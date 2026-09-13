#!/usr/bin/env python3
"""
Scraper léger dédié EXCLUSIVEMENT à la récupération d'emails d'agence.

Contrairement à scraper_finale_v2.py : pas de découpage par patterns
(contact/équipe/mentions), pas de nettoyage de texte, pas de cache de pages.
On crawle le site (même mini-crawler BFS, jusqu'à MAX_CRAWL_PAGES pages,
liens internes uniquement, footer compris puisqu'on ne touche jamais au
HTML), on extrait tous les emails trouvés sur chaque page visitée, et on
jette le HTML derrière. Rien d'autre n'est conservé.

Deux sources d'emails par page :
1. Regex classique sur le HTML brut (attrape le texte visible ET les hrefs
   mailto:).
2. Déchiffrement de l'obfuscation email Cloudflare (attribut
   data-cfemail="...") : Cloudflare remplace l'adresse réelle par un XOR
   hexadécimal + un span "[email protected]" visible. Sans ce décodage, ces
   adresses sont invisibles à un scraper classique.

Écrit dans emails_trouves_finale.jsonl (une ligne par agence : nom, site_web,
emails trouvés). Les agences sans email trouvé vont dans
agences_sans_email_trouve.jsonl (pour distinguer "site injoignable" de
"site OK mais vraiment aucun email dessus").

v2 : file d'attente continue au lieu de lots de taille fixe -- MAX_WORKERS
threads permanents piochent en continu dans une queue.Queue partagée. Un
domaine lent (même après le fix du timeout robots.txt) ne bloque plus QUE
son propre thread ; les autres continuent sans attendre. Reprise par nom
d'agence déjà présent dans emails_trouves_finale.jsonl /
agences_sans_email_trouve.jsonl (plus de progress.json à synchroniser).
Commit + push toutes les COMMIT_EVERY agences, s'arrête proprement avant la
limite de 6h de GitHub Actions.
"""

import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# --- Configuration ---------------------------------------------------------

CSV_PATH = Path("agences_sans_email_agence.csv")
RESULTS_PATH = Path("emails_trouves_finale.jsonl")
MISSING_PATH = Path("agences_sans_email_trouve.jsonl")

MAX_WORKERS = 8
REQUEST_DELAY = 0.3
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 1.5
MAX_CRAWL_PAGES = 12
COMMIT_EVERY = 50  # plus léger que le scraper de contenu -> commits moins fréquents
TIME_BUDGET_SECONDS = 5.5 * 3600

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
REQUEST_TIMEOUT = 10

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
CF_EMAIL_REGEX = re.compile(r'data-cfemail="([0-9a-fA-F]+)"')

# Faux positifs fréquents : extensions de fichiers prises pour un TLD par la
# regex (ex: "logo@2x.png" -> domaine "2x.png"), et domaines de tracking/
# placeholder qui ne sont jamais une vraie adresse de contact.
JUNK_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js",
             "json", "woff", "woff2", "ttf", "ico", "map"}
JUNK_DOMAIN_SUBSTRINGS = (
    "example.com", "example.org", "yourdomain", "domain.com",
    "sentry.io", "wixpress.com", "schema.org", "w3.org",
    "godaddy.com", "namecheap.com", "gandi.net",
)

ROBOTS_CACHE: dict[str, "RobotFileParser | None"] = {}
ROBOTS_LOCK = threading.Lock()
DOMAIN_LOCKS: dict[str, threading.Lock] = {}
DOMAIN_LOCKS_META_LOCK = threading.Lock()

_START_TIME = time.time()


def time_is_up() -> bool:
    return (time.time() - _START_TIME) >= TIME_BUDGET_SECONDS


# --- Lecture du CSV (tolérant aux apostrophes typographiques) --------------

def normalize_loose(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_agencies() -> list[dict]:
    if not CSV_PATH.exists():
        sys.exit(f"❌ {CSV_PATH} introuvable")
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        norm_to_actual = {normalize_loose(h): h for h in fieldnames}
        nom_col = norm_to_actual.get(normalize_loose("Nom de l'agence"))
        site_col = norm_to_actual.get(normalize_loose("Site web"))
        if not nom_col or not site_col:
            sys.exit(f"❌ Colonnes 'Nom de l'agence' / 'Site web' introuvables. "
                      f"Colonnes lues : {fieldnames}")
        rows = []
        for row in reader:
            rows.append({
                "nom": (row.get(nom_col) or "").strip(),
                "site_web": (row.get(site_col) or "").strip(),
            })
    return rows


# --- Réseau : robots.txt, verrou par domaine, retry -------------------------

def get_domain_lock(url: str) -> threading.Lock:
    domain = urlparse(url).netloc
    with DOMAIN_LOCKS_META_LOCK:
        return DOMAIN_LOCKS.setdefault(domain, threading.Lock())


def get_robots_parser(url: str) -> RobotFileParser | None:
    """Récupère et parse le robots.txt du domaine de `url`, avec cache par
    domaine. IMPORTANT : on ne laisse PAS RobotFileParser.read() faire sa
    propre requête réseau -- read() utilise urllib.request.urlopen() SANS
    AUCUN TIMEOUT, ce qui peut bloquer indéfiniment sur un serveur qui ne
    répond jamais (firewall silencieux, hébergeur capricieux). On récupère
    le contenu nous-mêmes via requests (timeout garanti) et on le donne à
    parse() à la place."""
    domain = urlparse(url).netloc
    with ROBOTS_LOCK:
        if domain in ROBOTS_CACHE:
            return ROBOTS_CACHE[domain]

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp: RobotFileParser | None = RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = requests.get(robots_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp = None  # pas de robots.txt (404 etc.) -> tout autorisé
    except requests.RequestException:
        rp = None  # injoignable -> on ne bloque pas le crawl pour ça

    with ROBOTS_LOCK:
        ROBOTS_CACHE[domain] = rp
    return rp


def _is_allowed(url: str, robots: RobotFileParser | None) -> bool:
    if robots is None:
        return True
    try:
        return robots.can_fetch(REQUEST_HEADERS["User-Agent"], url)
    except Exception:
        return True


def fetch(url: str, robots: RobotFileParser | None) -> requests.Response | None:
    if not _is_allowed(url, robots):
        return None
    for attempt in range(MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY)
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
                continue
            return None
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
            continue
        return None
    return None


def _same_domain(url: str, domain: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc.replace("www.", "") == domain.replace("www.", "")


# --- Extraction d'emails -----------------------------------------------------

def decode_cf_email(encoded: str) -> str | None:
    """Déchiffre l'obfuscation email Cloudflare (XOR simple)."""
    try:
        key = int(encoded[:2], 16)
        return "".join(
            chr(int(encoded[i:i + 2], 16) ^ key)
            for i in range(2, len(encoded), 2)
        )
    except Exception:
        return None


def is_junk_email(email: str) -> bool:
    _, _, domain_part = email.partition("@")
    if not domain_part:
        return True
    ext = domain_part.rsplit(".", 1)[-1]
    if ext in JUNK_TLDS:
        return True
    return any(j in domain_part for j in JUNK_DOMAIN_SUBSTRINGS)


def extract_emails(html: str) -> set[str]:
    found: set[str] = set()

    for m in CF_EMAIL_REGEX.finditer(html):
        decoded = decode_cf_email(m.group(1))
        if decoded:
            found.add(decoded.lower())

    for m in EMAIL_REGEX.finditer(html):
        found.add(m.group(0).lower())

    return {e for e in found if not is_junk_email(e)}


# --- Crawl + extraction, une agence -----------------------------------------

def process_agency(agency: dict) -> tuple[list[str] | None, str | None]:
    homepage_url = agency["site_web"]
    if not homepage_url:
        return None, "pas de site web"

    with get_domain_lock(homepage_url):
        robots = get_robots_parser(homepage_url)
        if not _is_allowed(homepage_url, robots):
            return None, "interdit par robots.txt"

        visited: set[str] = set()
        to_visit: list[str] = [homepage_url]
        emails: set[str] = set()
        domain: str | None = None
        any_page_fetched = False

        while to_visit and len(visited) < MAX_CRAWL_PAGES:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            resp = fetch(url, robots)
            if not resp:
                continue
            any_page_fetched = True

            if domain is None:
                domain = urlparse(resp.url).netloc

            emails |= extract_emails(resp.text)

            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("mailto:"):
                    addr = href[len("mailto:"):].split("?")[0].strip().lower()
                    if addr and not is_junk_email(addr):
                        emails.add(addr)
                    continue
                if not href or href.startswith(("tel:", "javascript:", "#")):
                    continue
                full_url = urljoin(resp.url, href).split("#")[0]
                if domain and not _same_domain(full_url, domain):
                    continue
                path_norm = normalize_loose(urlparse(full_url).path)
                if any(p in path_norm for p in ("wpadmin", "admin", "login", "panier", "cart", "logout")):
                    continue
                if full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

    if not any_page_fetched:
        return None, "site injoignable"
    if not emails:
        return None, "aucun email trouvé"
    return sorted(emails), None


# --- Reprise (par nom, plus par index) + commit git -------------------------

def load_already_done() -> set[str]:
    """Reprise basée sur les noms déjà présents dans les deux fichiers de
    sortie (succès ET échecs) -- plus de progress.json séparé à tenir
    synchronisé avec l'ordre réel de traitement (qui n'est plus garanti,
    vu que les threads terminent dans le désordre)."""
    done = set()
    for path in (RESULTS_PATH, MISSING_PATH):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nom = obj.get("nom")
                if nom:
                    done.add(nom)
    return done


def git_commit_and_push(message: str) -> None:
    try:
        subprocess.run(
            ["git", "add", str(RESULTS_PATH), str(MISSING_PATH)],
            check=True,
        )
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                return
            # Un commit qui échoue ici (identité git manquante, etc.) n'est
            # PAS transitoire : il échouera pareil à chaque prochaine
            # tentative. Continuer à traiter des agences sans pouvoir rien
            # sauvegarder reviendrait à brûler tout le budget de 5h30 pour
            # rien (déjà vécu une fois). On arrête tout de suite, fort et
            # clair, plutôt que de laisser filer jusqu'au bout silencieusement.
            print(f"❌ git commit a échoué de façon non transitoire :\n{result.stdout}\n{result.stderr}")
            sys.exit(1)
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ git add/push a échoué : {e}")
        sys.exit(1)


# --- Boucle principale : file d'attente continue -----------------------------

def worker(worker_id, work_queue, results_file, missing_file, write_lock,
           stats, stats_lock, commit_state, commit_lock, total_todo):
    """Boucle d'un thread permanent : dépile une agence de la file partagée,
    la traite, écrit le résultat, recommence -- jusqu'à file vide ou budget
    de temps atteint. Un domaine lent ne ralentit QUE ce thread."""
    while True:
        if time_is_up():
            return
        try:
            agency = work_queue.get_nowait()
        except queue.Empty:
            return

        name = agency["nom"] or "(sans nom)"
        emails, reason = process_agency(agency)

        with write_lock:
            if emails:
                results_file.write(json.dumps(
                    {"nom": name, "site_web": agency["site_web"], "emails": emails},
                    ensure_ascii=False,
                ) + "\n")
                results_file.flush()
            else:
                missing_file.write(json.dumps(
                    {"nom": name, "site_web": agency["site_web"], "raison": reason},
                    ensure_ascii=False,
                ) + "\n")
                missing_file.flush()

        with stats_lock:
            if emails:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            done_count = stats["success"] + stats["failed"]

        if emails:
            print(f"[{done_count}/{total_todo}] {name} -> {len(emails)} email(s) : {', '.join(emails)}")
        else:
            print(f"[{done_count}/{total_todo}] {name} -> {reason}")

        with commit_lock:
            commit_state["since_commit"] += 1
            if commit_state["since_commit"] >= COMMIT_EVERY:
                git_commit_and_push(f"Progression extraction emails : {done_count}/{total_todo} ce run")
                commit_state["since_commit"] = 0

        work_queue.task_done()


def main() -> None:
    agencies = load_agencies()
    already_done = load_already_done()
    total = len(agencies)
    todo = [a for a in agencies if (a["nom"] or "(sans nom)") not in already_done]

    print(f"{total} agence(s) au total, {len(already_done)} déjà traitée(s), "
          f"{len(todo)} restante(s) — {MAX_WORKERS} threads permanents")

    if not todo:
        print("Toutes les agences ont déjà été traitées.")
        return

    work_queue: "queue.Queue" = queue.Queue()
    for a in todo:
        work_queue.put(a)

    results_file = RESULTS_PATH.open("a", encoding="utf-8")
    missing_file = MISSING_PATH.open("a", encoding="utf-8")
    write_lock = threading.Lock()
    stats = {"success": 0, "failed": 0}
    stats_lock = threading.Lock()
    commit_state = {"since_commit": 0}
    commit_lock = threading.Lock()

    try:
        threads = [
            threading.Thread(
                target=worker,
                args=(i, work_queue, results_file, missing_file, write_lock,
                      stats, stats_lock, commit_state, commit_lock, len(todo)),
                daemon=True,
            )
            for i in range(MAX_WORKERS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        results_file.close()
        missing_file.close()
        if commit_state["since_commit"] > 0:
            git_commit_and_push("Progression extraction emails : checkpoint final du run")

    leftover = work_queue.qsize()
    if leftover > 0:
        print(f"\n⏱️ Budget temps atteint (5h30) — {leftover} agence(s) restée(s) dans la file, "
              f"reprises automatiquement au prochain run.")
    else:
        print(f"\n✅ Run terminé : {stats['success']} succès, {stats['failed']} sans email/injoignable.")


if __name__ == "__main__":
    main()
