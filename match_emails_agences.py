#!/usr/bin/env python3
"""Fusion des emails scrapés et matching gérant/email avec Cerebras.

Le fonctionnement Cerebras reprend le modèle de l'extracteur existant :
- une file partagée d'agences ambiguës ;
- un thread dédié par clé API ;
- un KeyRateLimiter indépendant par clé (TPM + RPM) ;
- les 429 de débit sont retentés après attente ;
- une clé invalide ou à quota journalier épuisé est retirée, et l'agence en
  cours est remise dans la file pour une autre clé ;
- le résultat final conserve l'ordre du CSV source.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_TPM_LIMIT = 30_000
DEFAULT_RPM_LIMIT = 5
MAX_COMPLETION_TOKENS = 250
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
GENERIC_PREFIXES = {
    "contact", "info", "accueil", "agence", "agences", "bonjour", "hello",
    "admin", "direction", "secretariat", "standard", "commercial", "location",
    "locations", "vente", "ventes", "transaction", "gestion", "rh", "recrutement",
    "service", "support", "office", "team", "equipe", "immo", "immobilier",
    "property", "assistance", "compta", "comptabilite",
}
GENERIC_PRIORITY = {"contact": 0, "info": 1, "accueil": 2, "agence": 3, "bonjour": 4, "hello": 5}
STOP = {"m", "mme", "mr", "mlle", "monsieur", "madame", "me", "dr", "le", "la", "de", "du", "des", "d", "et"}
# Domaines mail publics : ne jamais indexer par ce domaine, sinon deux agences
# distinctes utilisant toutes deux une adresse @gmail.com (par ex.) se
# retrouveraient à partager leurs emails via cette clé d'index.
PUBLIC_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "hotmail.fr", "outlook.com", "outlook.fr",
    "yahoo.com", "yahoo.fr", "orange.fr", "wanadoo.fr", "free.fr", "sfr.fr",
    "laposte.net", "icloud.com", "live.fr", "live.com", "bbox.fr", "neuf.fr",
    "aol.com", "gmx.fr", "gmx.com", "protonmail.com",
}
CHECKPOINT_EVERY = 20  # écrit le JSONL de sortie tous les N résultats traités par l'IA


class KeyDeadError(Exception):
    pass


class DailyQuotaExhausted(KeyDeadError):
    pass


class KeyRateLimiter:
    """Limiteur indépendant : aucune clé ne partage son compteur avec une autre."""
    def __init__(self, tpm_limit: int, rpm_limit: int):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.tokens: list[tuple[float, int]] = []
        self.requests: list[float] = []
        self.lock = threading.Lock()

    def _purge(self, now: float) -> None:
        self.tokens = [(t, n) for t, n in self.tokens if now - t <= 60]
        self.requests = [t for t in self.requests if now - t <= 60]

    def wait_if_needed(self, estimated_tokens: int) -> None:
        while True:
            with self.lock:
                now = time.time()
                self._purge(now)
                used = sum(n for _, n in self.tokens)
                waits = []
                if used + estimated_tokens > self.tpm_limit and self.tokens:
                    waits.append(60 - (now - self.tokens[0][0]) + 0.5)
                if self.rpm_limit and len(self.requests) >= self.rpm_limit:
                    waits.append(60 - (now - self.requests[0]) + 0.5)
                wait = max(waits, default=0)
            if wait <= 0:
                return
            print(f"⏱️ clé Cerebras : pause {wait:.1f}s (limite TPM/RPM)", flush=True)
            time.sleep(wait)

    def record(self, tokens: int) -> None:
        with self.lock:
            now = time.time()
            self.tokens.append((now, max(1, tokens)))
            self.requests.append(now)


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def words(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return [x for x in re.split(r"[^a-z0-9]+", text) if x and x not in STOP]


def clean_email(value: Any) -> str:
    value = str(value or "").strip().lower().strip("<>[](){};,\"'")
    return value if EMAIL_RE.match(value) else ""


def is_generic(email: str) -> bool:
    local = email.split("@", 1)[0].lower()
    base = re.split(r"[+._-]", local, maxsplit=1)[0]
    return local in GENERIC_PREFIXES or base in GENERIC_PREFIXES or any(
        local.startswith(p + sep) for p in GENERIC_PREFIXES for sep in (".", "-", "_")
    )


def generic_key(email: str) -> tuple[int, str]:
    base = re.split(r"[+._-]", email.split("@", 1)[0].lower(), maxsplit=1)[0]
    return GENERIC_PRIORITY.get(base, 50), email


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"⚠️ ligne JSONL {line_no} ignorée : {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict): result.append(obj)
    return result


def index_scraped(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, set[str]] = {}
    for row in rows:
        raw = row.get("emails", [])
        if isinstance(raw, str): raw = re.split(r"[,;\n]", raw)
        emails = {e for e in (clean_email(x) for x in raw) if e}
        keys = [norm(row.get("nom")), norm(row.get("site_web"))]
        for email in emails:
            domain = email.split("@", 1)[1]
            if domain not in PUBLIC_EMAIL_DOMAINS:
                keys.append(domain)
        for key in keys:
            if key: index.setdefault(key, set()).update(emails)
    return {k: sorted(v) for k, v in index.items()}


def candidates_for(row: dict[str, str], index: dict[str, list[str]]) -> list[str]:
    site = (row.get("Site web") or "").lower().replace("https://", "").replace("http://", "")
    domain = re.sub(r"^www\.", "", site.split("/", 1)[0])
    values: set[str] = set()
    for key in (norm(row.get("Nom de l’agence")), norm(row.get("Nom de l'agence")), norm(row.get("Site web")), domain):
        values.update(index.get(key, []))
    return sorted(values)


def score_email(manager: str, email: str) -> int:
    if not manager or not email or is_generic(email): return -100
    local = norm(email.split("@", 1)[0])
    parts = words(manager)
    score = sum(4 for p in parts if len(p) >= 2 and p in local)
    compact = norm(manager)
    if len(compact) >= 4 and compact in local: score += 8
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        if local.startswith(first): score += 2
        if last in local: score += 3
        if local.startswith(last): score += 2
        if first[:1] + last in local or last + first[:1] in local: score += 5
    return score


def heuristic_match(manager: str, emails: list[str]) -> tuple[str, int, str]:
    nominative = [e for e in emails if not is_generic(e)]
    if not nominative: return "", 0, "none"
    ranked = sorted(((score_email(manager, e), e) for e in nominative), reverse=True)
    best_score, best = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else -999
    if best_score >= 5 and best_score > second: return best, best_score, "heuristic"
    if len(nominative) == 1 and best_score >= 0: return best, best_score, "single_nominative"
    return "", best_score, "ambiguous"


def estimate_tokens(manager: str, emails: list[str]) -> int:
    return max(300, (len(manager) + sum(len(e) for e in emails)) // 3 + 250)


SYSTEM_PROMPT = """Tu fais du rapprochement entre le nom d'un gérant d'agence immobilière et une liste d'adresses email nominatives candidates (les adresses génériques ont déjà été écartées en amont).

Règles pour juger une correspondance fiable, par ordre de force :
- le nom complet du gérant (sans espaces/accents) apparaît dans la partie locale de l'email (avant le @) ;
- l'initiale du prénom suivie du nom de famille, ou le nom de famille suivi de l'initiale du prénom, apparaît dans la partie locale (ex. "Jean Dupont" -> "jdupont" ou "dupontj") ;
- le nom de famille seul apparaît dans la partie locale, sans ambiguïté avec un autre candidat.

Un prénom seul, une simple ressemblance phonétique, ou une correspondance partielle sur un nom très courant (Martin, Bernard...) ne suffisent pas : dans le doute, ne choisis rien plutôt que de deviner.

S'il y a plusieurs candidats plausibles et qu'aucun ne se détache clairement des autres selon ces règles, ne choisis rien.

Réponds uniquement avec un objet JSON de la forme {"email": "", "reason": ""} : "email" est l'adresse choisie parmi les candidats fournis (chaîne vide si aucune correspondance fiable), "reason" est une justification en une courte phrase, en français."""


def cerebras_choose(manager: str, emails: list[str], api_key: str, model: str, limiter: KeyRateLimiter) -> tuple[str, str, int]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "gerant": manager,
                "emails_candidats": emails,
            }, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
    }
    estimated = estimate_tokens(manager, emails)
    limiter.wait_if_needed(estimated)
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=60)
        except requests.exceptions.RequestException as exc:
            # Blip réseau (timeout, connexion coupée, DNS...) : on retente comme
            # pour un 500, on ne tue pas la clé pour ça.
            if attempt < max_attempts:
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"Cerebras : erreur réseau après {max_attempts} tentatives : {exc}") from exc
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage") or {}
            limiter.record(int(usage.get("total_tokens") or estimated))
            text = re.sub(r"```json|```", "", data["choices"][0]["message"]["content"]).strip()
            obj = json.loads(text)
            chosen = clean_email(obj.get("email"))
            if chosen not in emails or is_generic(chosen): chosen = ""
            return chosen, str(obj.get("reason", "")), int(usage.get("total_tokens") or estimated)
        detail = response.text[:300].replace("\n", " ")
        if response.status_code == 429:
            day_tokens = response.headers.get("x-ratelimit-remaining-tokens-day")
            day_requests = response.headers.get("x-ratelimit-remaining-requests-day")
            if day_tokens is not None and day_tokens.isdigit() and int(day_tokens) <= 0:
                raise DailyQuotaExhausted(detail)
            if day_requests is not None and day_requests.isdigit() and int(day_requests) <= 0:
                raise DailyQuotaExhausted(detail)
            if attempt < max_attempts:
                time.sleep(_retry_delay(response, default=10 * attempt))
                continue
            raise KeyDeadError(f"429 après retries : {detail}")
        if response.status_code in (401, 403) or (response.status_code == 400 and "organization_restricted" in detail):
            raise KeyDeadError(f"clé refusée : {response.status_code} {detail}")
        if response.status_code in (500, 502, 503) and attempt < max_attempts:
            time.sleep(_retry_delay(response, default=5 * attempt))
            continue
        raise RuntimeError(f"Cerebras {response.status_code}: {detail}")
    raise RuntimeError("Cerebras : retries épuisés")


def _retry_delay(response: "requests.Response", default: float) -> float:
    """Utilise Retry-After ou le header de reset Cerebras si présent, sinon le backoff par défaut."""
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.5)
        except ValueError:
            pass
    reset = response.headers.get("x-ratelimit-reset-tokens-minute") or response.headers.get("x-ratelimit-reset-requests-minute")
    if reset is not None:
        try:
            return max(float(reset) + 0.5, 0.5)
        except ValueError:
            pass
    return default


def base_result(row: dict[str, str], index: dict[str, list[str]]) -> tuple[dict[str, Any], bool]:
    emails = candidates_for(row, index)
    generic = sorted((e for e in emails if is_generic(e)), key=generic_key)
    manager = (row.get("Gérant") or "").strip()
    existing = clean_email(row.get("Email du gérant"))
    selected, score, method = (existing, 999, "existing_csv") if existing else heuristic_match(manager, emails)
    needs_ai = not existing and method == "ambiguous" and bool([e for e in emails if not is_generic(e)])
    result = {
        "Agence": (row.get("Nom de l’agence") or row.get("Nom de l'agence") or "").strip(),
        "E-mail agence": generic[0] if generic else "",
        "E-mails agence generiques": generic,
        "E-mail gérant": selected,
        "Gérant": manager,
        "tous_les_emails_trouves": emails,
        "matching": {"methode": method, "score": score, "note": ""},
        "matched_at": datetime.now(timezone.utc).isoformat(),
    }
    return result, needs_ai


def write_output(path: Path, results: list[dict[str, Any]]) -> None:
    """Écriture atomique (fichier temporaire + rename) pour ne jamais laisser
    un JSONL de sortie tronqué si le job est interrompu pendant l'écriture."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for result in results:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    tmp.replace(path)


def multi_key_ai(results: list[dict[str, Any]], tasks: list[tuple[int, str, list[str]]], keys: list[str], model: str, tpm: int, rpm: int, output_path: Path | None = None) -> None:
    if not tasks or not keys:
        return
    work: queue.Queue[tuple[int, str, list[str]]] = queue.Queue()
    for task in tasks: work.put(task)
    dead: set[int] = set()
    dead_lock = threading.Lock()
    result_lock = threading.Lock()
    completed = 0

    def maybe_checkpoint() -> None:
        # Appelé sous result_lock : écrit un instantané de results tous les
        # CHECKPOINT_EVERY résultats, pour ne pas tout perdre si le job est
        # interrompu (timeout GitHub Actions, crash, etc.).
        nonlocal completed
        completed += 1
        if output_path is not None and completed % CHECKPOINT_EVERY == 0:
            write_output(output_path, results)
            print(f"💾 checkpoint : {completed} résultat(s) IA écrits dans {output_path}", flush=True)

    def worker(key_idx: int, api_key: str) -> None:
        limiter = KeyRateLimiter(tpm, rpm)
        while True:
            try:
                idx, manager, emails = work.get(timeout=0.4)
            except queue.Empty:
                return
            try:
                chosen, reason, used = cerebras_choose(manager, emails, api_key, model, limiter)
                with result_lock:
                    results[idx]["E-mail gérant"] = chosen
                    results[idx]["matching"] = {"methode": "cerebras" if chosen else "unmatched", "score": 0, "note": reason}
                    maybe_checkpoint()
                print(f"[{key_idx + 1}/{len(keys)}] {results[idx]['Agence']} -> {chosen or '(aucune correspondance)'}", flush=True)
                work.task_done()
            except KeyDeadError as exc:
                with dead_lock: dead.add(key_idx)
                work.put((idx, manager, emails))
                work.task_done()
                print(f"⚠️ clé #{key_idx + 1} retirée ({exc}); agence remise en file", flush=True)
                return
            except Exception as exc:
                with result_lock:
                    results[idx]["matching"] = {"methode": "cerebras_error", "score": 0, "note": str(exc)}
                    maybe_checkpoint()
                work.task_done()

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = [pool.submit(worker, i, key) for i, key in enumerate(keys)]
        for future in futures: future.result()
    while not work.empty():
        idx, _, _ = work.get_nowait()
        with result_lock:
            results[idx]["matching"] = {"methode": "unmatched_no_live_key", "score": 0, "note": "Toutes les clés Cerebras disponibles ont été épuisées."}
        work.task_done()
    if dead: print(f"Clés retirées: {len(dead)}/{len(keys)}; tâches restantes traitées sans clé vivante.", flush=True)
    if output_path is not None:
        write_output(output_path, results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agencies", default="agences_sans_email_agence.csv")
    parser.add_argument("--scraped", default="emails_trouves_finale.jsonl")
    parser.add_argument("--output", default="agences_emails_matchees.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--model", default=os.getenv("CEREBRAS_MODEL", DEFAULT_MODEL))
    parser.add_argument("--tpm-limit", type=int, default=int(os.getenv("CEREBRAS_TPM_LIMIT", DEFAULT_TPM_LIMIT)))
    parser.add_argument("--rpm-limit", type=int, default=int(os.getenv("CEREBRAS_RPM_LIMIT", DEFAULT_RPM_LIMIT)))
    args = parser.parse_args()
    agencies_path, scraped_path = Path(args.agencies), Path(args.scraped)
    if not agencies_path.exists(): raise SystemExit(f"Fichier introuvable: {agencies_path}")
    if not scraped_path.exists(): raise SystemExit(f"Fichier introuvable: {scraped_path}")
    rows = load_csv(agencies_path)
    if args.limit: rows = rows[:args.limit]
    index = index_scraped(load_jsonl(scraped_path))
    results, tasks = [], []
    for idx, row in enumerate(rows):
        result, needs_ai = base_result(row, index)
        results.append(result)
        if needs_ai:
            tasks.append((idx, result["Gérant"], [e for e in result["tous_les_emails_trouves"] if not is_generic(e)]))
    output_path = Path(args.output)
    # Checkpoint immédiat du matching déterministe (avant tout appel IA) :
    # si le job est interrompu pendant les appels Cerebras, ce résultat n'est
    # jamais perdu.
    write_output(output_path, results)
    raw_keys = os.getenv("CEREBRAS_API_KEYS", "") or os.getenv("CEREBRAS_API_KEY", "")
    keys = [k.strip() for k in re.split(r"[,\n]", raw_keys) if k.strip()]
    if tasks and not args.no_ai:
        if not keys: print("⚠️ aucune clé Cerebras : les cas ambigus restent non rapprochés", file=sys.stderr)
        else: multi_key_ai(results, tasks, keys, args.model, args.tpm_limit, args.rpm_limit, output_path)
    write_output(output_path, results)
    print(f"Terminé: {len(rows)} agences, {len(tasks)} cas ambigus, {len(keys)} clé(s) Cerebras détectée(s) -> {args.output}")


if __name__ == "__main__": main()
