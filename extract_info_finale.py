#!/usr/bin/env python3
"""
ÉTAPE 2 : lit les pages nettoyées dans pages_cache_finale.jsonl (produites
par scraper_finale_v2.py) et utilise GPT-OSS-120B (via Cerebras) pour en
extraire, AGENCE PAR AGENCE, les infos suivantes :
  - Adresse
  - Téléphone de l'agence
  - Email de l'agence
  - Horaires
  - Gérant (nom)
  - Téléphone du gérant
  - Email du gérant

Variante "finale" : lit DIRECTEMENT pages_cache_finale.jsonl (nom, site_web,
pages), sans passer par un CSV intermédiaire — le cache contient déjà tout
ce qu'il faut. Écrit dans extraction_results_finale.jsonl, séparément des
autres runs (principal / refair) pour ne rien mélanger.

Chaque ligne de sortie contient 11 clés :
  nom, site_web, extracted_at, adresse, telephone_agence, email_agence,
  horaires, gerant, tel_gerant, email_gerant, pages_sources

BOOST v2 (par rapport à extract_info_refair.py) :
  - Parallélisation réelle : un thread DÉDIÉ par clé API Cerebras (pas de
    va-et-vient round-robin sur un seul thread comme avant). Avec N clés
    vivantes, jusqu'à N agences sont extraites EN MÊME TEMPS, chaque thread
    respectant le budget TPM/RPM de SA PROPRE clé (pas de contention/lock
    sur le rate limiter, chaque limiter n'est touché que par un seul thread).
  - Si une clé meurt (quota jour épuisé, clé invalide), son thread s'arrête
    et remet l'agence en cours dans la file pour qu'un thread encore vivant
    la retente avec sa propre clé — aucun travail perdu.
  - Plus besoin de CSV/colonne "Nom de l'agence" à faire correspondre : la
    source de vérité est directement pages_cache_finale.jsonl.

MISE À JOUR RATE LIMITING (Cerebras : TPM ET RPM par clé) :
  - Chaque thread (= chaque clé) a son propre KeyRateLimiter à fenêtre
    glissante de 60s, qui suit tokens réellement consommés (champ "usage"
    renvoyé par Cerebras) et nombre de requêtes, et attend automatiquement
    avant une requête qui dépasserait le budget de SA clé.
  - Si le contenu d'une agence est trop volumineux pour tenir dans une
    seule requête, il est découpé en plusieurs morceaux (par page, et si
    besoin une page est elle-même coupée). Chaque morceau est envoyé
    séparément, l'IA est prévenue qu'il s'agit d'une partie du site d'une
    même agence. Les résultats des morceaux sont ensuite fusionnés (premier
    champ non vide retenu, source de page conservée).
"""

import json
import os
import queue
import re
import sys
import threading
import time
import argparse
import collections
from datetime import datetime, timezone

import requests

CACHE_PATH = "pages_cache_finale.jsonl"
RESULTS_PATH = "extraction_results_finale.jsonl"

MODEL = "gpt-oss-120b"
API_URL = "https://api.cerebras.ai/v1/chat/completions"
API_KEYS_ENV_VAR = "CEREBRAS_API_KEYS"

# --- Rate limiting Cerebras (PAR CLÉ) -----------------------------------
# ⚠️ Ce sont des valeurs par défaut prudentes. Cerebras limite à la fois
# le débit en tokens/minute (TPM) ET en requêtes/minute (RPM) — contrairement
# à Groq où seul le TPM posait problème. Vérifie les vraies valeurs de TON
# compte sur https://cloud.cerebras.ai/platform/.../limits et ajuste avec
# --tpm-limit / --rpm-limit si besoin (pas la peine d'éditer ce fichier).
TPM_LIMIT_DEFAULT = 30000   # tokens NON CACHÉS/minute, PAR CLÉ (confirmé sur le dashboard)
RPM_LIMIT_DEFAULT = 5       # requêtes/minute, PAR CLÉ (confirmé sur le dashboard)
# Plafonds journaliers PAR CLÉ, confirmés sur le dashboard (informatif — pas
# encore appliqués par le limiteur, qui ne suit que la fenêtre de 60s) :
#   - 2 400 requêtes/jour
#   - 1 000 000 tokens non cachés/jour
#   - 3 000 000 tokens au total/jour
# Avec 8 clés : ~19 200 requêtes/jour et ~8 000 000 tokens non cachés/jour au
# total. Si le volume de texte à traiter dépasse ce budget journalier, les
# clés sembleront "mortes" (429 persistants) jusqu'au reset du quota le
# lendemain — ce n'est pas une panne, le script reprendra automatiquement
# via already_done au run suivant.

# gpt-oss-120b est un modèle "reasoning" : une partie de la complétion est
# consommée par le raisonnement interne AVANT le JSON final. Avec
# reasoning_effort=low ce raisonnement reste léger, mais on garde quand
# même une marge confortable pour ne jamais tronquer le JSON en sortie.
COMPLETION_TOKENS_RESERVED = 1500  # tokens réservés (raisonnement + JSON)
MAX_TOKENS_PARAM = 1500     # plafond envoyé à l'API pour la complétion
REASONING_EFFORT = "low"    # vrai paramètre API (pas juste une instruction texte)
CHARS_PER_TOKEN = 4         # approximation grossière (~0.75 mot/token en FR)
# Taille max d'un morceau de contenu "pages" envoyé en une seule requête,
# pour laisser de la marge au prompt système/instructions + à la réponse
# dans le budget de tokens/minute d'une clé.
MAX_CHUNK_TOKENS = 3000
PROMPT_OVERHEAD_TOKENS = 700  # système + instructions + schéma JSON (approx)


class KeyDeadError(Exception):
    pass


class DailyQuotaExhausted(KeyDeadError):
    """Clé bloquée pour la journée (quota jour Cerebras épuisé), par
    opposition à une clé vraiment invalide (401/403) ou à un 429 passager
    de débit minute. Se réinitialise au reset quotidien du fournisseur."""
    pass


def estimate_tokens(text):
    """Estimation grossière : ~4 caractères par token."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def load_api_keys(env_var):
    raw = os.environ.get(env_var, "")
    keys = [k.strip() for k in re.split(r"[,\n]", raw) if k.strip()]
    if not keys:
        sys.exit(f"Aucune clé API trouvée dans la variable d'env {env_var}")
    return keys


def load_agencies():
    """Lit pages_cache_finale.jsonl directement : c'est la seule source de
    vérité (plus de CSV à faire correspondre). Renvoie une liste ordonnée
    de {"nom", "site_web", "pages"} (pages déjà normalisées)."""
    if not os.path.exists(CACHE_PATH):
        sys.exit(f"❌ {CACHE_PATH} introuvable — lance d'abord scraper_finale_v2.py")
    agencies = []
    seen_noms = set()
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            nom = obj.get("nom", "").strip()
            if not nom or nom in seen_noms:
                continue  # doublon éventuel après fusion de caches : on garde la première occurrence
            seen_noms.add(nom)
            agencies.append({
                "nom": nom,
                "site_web": obj.get("site_web", "").strip(),
                "pages": normalize_pages(obj.get("pages", [])),
            })
    return agencies


def normalize_pages(pages_raw):
    """Normalise "pages" (dict, liste de dicts, ou liste de chaînes) en
    une liste d'objets {"url": ..., "text": ...}."""
    normalized = []

    if isinstance(pages_raw, dict):
        for url, text in pages_raw.items():
            if isinstance(text, dict):
                normalized.append({"url": text.get("url", url), "text": text.get("text", "")})
            else:
                normalized.append({"url": url, "text": text if isinstance(text, str) else ""})
        return normalized

    for p in pages_raw or []:
        if isinstance(p, dict):
            normalized.append({"url": p.get("url", ""), "text": p.get("text", "")})
        elif isinstance(p, str):
            normalized.append({"url": p, "text": ""})
    return normalized


def load_existing_results():
    done = set()
    if not os.path.exists(RESULTS_PATH):
        return done
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                nom = obj.get("nom", "")
                if nom:
                    done.add(nom)
            except json.JSONDecodeError:
                continue
    return done


SYSTEM_MSG = (
    "Tu es un assistant précis spécialisé dans l'extraction d'informations "
    "de contact à partir de pages web d'agences immobilières françaises.\n"
    "Reasoning: low.\n"
    "Règles générales :\n"
    "- Réponds uniquement à partir du texte fourni, n'invente jamais une information absente.\n"
    "- Si une info n'est pas présente dans le texte, mets une chaîne vide \"\".\n"
    "- Distingue bien les coordonnées de L'AGENCE (souvent contact@, info@, standard) "
    "de celles du GÉRANT/DIRECTEUR (souvent prénom.nom@, mentionné nommément).\n"
    "- Réponds UNIQUEMENT avec un objet JSON valide, sans balises markdown ni texte autour."
)


def build_user_prompt(nom, chunk_items, chunk_index=None, chunk_total=None):
    """chunk_items : liste de (label, url, text) où label est le numéro de
    page d'origine (ex: "3" ou "3b" si une page a été redécoupée)."""
    if not chunk_items:
        pages_block = "(aucune page récupérée)"
    else:
        blocks = []
        for label, url, text in chunk_items:
            blocks.append(f"--- Page {label} ({url}) ---\n{text}")
        pages_block = "\n\n".join(blocks)

    chunk_note = ""
    if chunk_total and chunk_total > 1:
        chunk_note = (
            f"\n## Remarque importante\n\n"
            f"Ceci est la partie {chunk_index}/{chunk_total} du contenu du site de "
            f"l'agence \"{nom}\" (le site a été découpé car trop volumineux pour une "
            f"seule requête). D'autres parties de ce même site existent ailleurs : si une "
            f"information demandée n'apparaît pas dans CE texte, laisse le champ "
            f"correspondant vide (\"\") plutôt que d'inventer — elle sera peut-être "
            f"trouvée dans une autre partie.\n"
        )

    return f"""## Contexte

Voici le texte nettoyé de pages extraites du site web de l'agence immobilière "{nom}".
{chunk_note}
## Tâche

Extrait les informations suivantes à partir de ce texte uniquement :
- adresse postale de l'agence
- téléphone de l'agence (numéro générique/standard)
- email de l'agence (générique : contact@, info@, agence@, accueil@...)
- horaires d'ouverture de l'agence
- nom de la personne retenue comme "gérant" selon la règle de sélection ci-dessous
- téléphone de cette même personne (si un numéro distinct lui est attribué nommément)
- email de cette même personne (nominatif : prénom.nom@... ou similaire)

## Règle de sélection de la personne à retenir comme "gérant"

Le site peut citer plusieurs personnes avec un titre de responsable : directeur,
gérant, responsable d'agence, directeur adjoint, etc. Applique cette règle de
priorité, DANS L'ORDRE :

1. Si le directeur/gérant PRINCIPAL de l'agence a un email nominatif (du type
   prenom.nom@..., ou toute adresse clairement personnelle, pas générique),
   retiens SES informations (nom, téléphone, email) comme "gérant".
2. Sinon (le directeur principal n'a pas d'email nominatif), mais qu'une AUTRE
   personne de l'agence portant un titre de responsable (responsable d'agence,
   directeur adjoint, un autre directeur, etc.) a elle un email nominatif :
   retiens LES INFORMATIONS DE CETTE AUTRE PERSONNE (nom, téléphone, email)
   comme "gérant" — même si son titre n'est pas "directeur".
3. Sinon (aucune personne trouvée n'a d'email nominatif) : retiens quand même
   le directeur/gérant principal de l'agence, même si son email est générique
   ou absent.

Important : les champs "gerant", "tel_gerant" et "email_gerant" doivent TOUS
LES TROIS correspondre à LA MÊME personne (celle retenue par la règle
ci-dessus). Ne mélange jamais le nom d'une personne avec l'email ou le
téléphone d'une autre.

## Contraintes

- N'invente aucune information absente du texte.
- Pour chaque champ "gérant", indique aussi la page source (le numéro "Page N" exact où l'info a été trouvée), ou "" si non trouvé.

## Format de sortie

Retourne uniquement un JSON valide correspondant exactement à ce schéma :

{{
  "adresse": "",
  "telephone_agence": "",
  "email_agence": "",
  "horaires": "",
  "gerant": {{"valeur": "", "page": ""}},
  "tel_gerant": {{"valeur": "", "page": ""}},
  "email_gerant": {{"valeur": "", "page": ""}}
}}

## Pages du site

{pages_block}
"""


def split_text_into_pieces(text, max_tokens):
    """Découpe un texte trop long en morceaux de ~max_tokens tokens,
    sans couper au milieu d'un mot."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    words = text.split()
    pieces = []
    current = []
    current_len = 0
    for w in words:
        wlen = len(w) + 1
        if current_len + wlen > max_chars and current:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(w)
        current_len += wlen
    if current:
        pieces.append(" ".join(current))
    return pieces or [""]


def build_chunks(pages, max_chunk_tokens):
    """Regroupe les pages d'une agence en morceaux dont le contenu tient
    dans max_chunk_tokens. Une page individuellement trop grosse est
    elle-même redécoupée (labels "Na", "Nb", ...). Renvoie une liste de
    chunks, chaque chunk étant une liste de (label, url, text)."""
    chunks = []
    current = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(current)
            current = []
            current_tokens = 0

    for idx, p in enumerate(pages, start=1):
        text = p.get("text", "") or ""
        url = p.get("url", "")
        ptoks = estimate_tokens(text)

        if ptoks <= max_chunk_tokens:
            if current_tokens + ptoks > max_chunk_tokens:
                flush()
            current.append((str(idx), url, text))
            current_tokens += ptoks
        else:
            # Cette page seule dépasse le budget d'un chunk : on la redécoupe.
            pieces = split_text_into_pieces(text, max_chunk_tokens)
            for j, piece in enumerate(pieces):
                label = f"{idx}{chr(97 + j)}" if len(pieces) > 1 else str(idx)
                piece_toks = estimate_tokens(piece)
                if current_tokens + piece_toks > max_chunk_tokens:
                    flush()
                current.append((label, url, piece))
                current_tokens += piece_toks

    flush()
    return chunks or [[]]


def resolve_page_url(pages, page_ref):
    """Convertit une référence 'Page N' (ou 'Page Nb') en URL réelle."""
    if not page_ref:
        return ""
    m = re.search(r"\d+", str(page_ref))
    if not m:
        return ""
    idx = int(m.group()) - 1
    if 0 <= idx < len(pages):
        return pages[idx]["url"]
    return ""


class KeyRateLimiter:
    """Limiteur de débit par clé : fenêtre glissante de 60s à la fois sur
    les tokens réellement consommés (via le champ 'usage' renvoyé par
    l'API) ET sur le nombre de requêtes envoyées (Cerebras limite les
    deux — contrairement à Groq où seul le TPM posait problème)."""

    def __init__(self, tpm_limit=TPM_LIMIT_DEFAULT, rpm_limit=RPM_LIMIT_DEFAULT):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.token_usage = collections.deque()    # (timestamp, tokens)
        self.request_usage = collections.deque()  # (timestamp,)

    def _purge(self, now):
        while self.token_usage and now - self.token_usage[0][0] > 60:
            self.token_usage.popleft()
        while self.request_usage and now - self.request_usage[0] > 60:
            self.request_usage.popleft()

    def wait_if_needed(self, estimated_tokens):
        now = time.time()
        self._purge(now)

        wait_time = 0.0
        reason = ""

        tokens_used = sum(t for _, t in self.token_usage)
        if tokens_used + estimated_tokens > self.tpm_limit:
            if self.token_usage:
                wait_time = max(wait_time, 60 - (now - self.token_usage[0][0]) + 0.5)
            else:
                wait_time = max(wait_time, 1.0)
            reason = f"~{self.tpm_limit} tokens/min"

        if self.rpm_limit and len(self.request_usage) >= self.rpm_limit:
            wait_time = max(wait_time, 60 - (now - self.request_usage[0]) + 0.5)
            reason = f"{self.rpm_limit} requêtes/min" if not reason else reason + f" et {self.rpm_limit} requêtes/min"

        if wait_time > 0:
            print(f"    ⏱️ Limite Cerebras ({reason}) proche pour cette clé, pause {wait_time:.1f}s...")
            time.sleep(wait_time)
            self._purge(time.time())

    def record(self, tokens):
        now = time.time()
        self.token_usage.append((now, tokens))
        self.request_usage.append(now)


def call_cerebras(nom, chunk_items, api_key, chunk_index=None, chunk_total=None, max_retries=3):
    """Envoie un chunk à Cerebras. Renvoie (texte_reponse, usage_dict|None)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": build_user_prompt(nom, chunk_items, chunk_index, chunk_total)},
        ],
        "temperature": 0.1,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_TOKENS_PARAM,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = None
    for attempt in range(1, max_retries + 1):
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 503 and attempt < max_retries:
            wait = 5 * attempt
            print(f"    ⏳ 503 (tentative {attempt}/{max_retries}), retry dans {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 429 and attempt < max_retries:
            # Rate limit dépassé malgré le limiteur (marge de sécurité) : on attend un peu.
            wait = 10 * attempt
            print(f"    ⏳ 429 (tentative {attempt}/{max_retries}), retry dans {wait}s...")
            time.sleep(wait)
            continue
        break

    if resp.status_code == 200:
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            text = None
        usage = data.get("usage")
        return text, usage

    detail = resp.text[:300].replace("\n", " ")

    if resp.status_code == 429:
        remaining_tokens_day = resp.headers.get("x-ratelimit-remaining-tokens-day")
        remaining_requests_day = resp.headers.get("x-ratelimit-remaining-requests-day")
        day_exhausted = False
        try:
            if remaining_tokens_day is not None and float(remaining_tokens_day) <= 0:
                day_exhausted = True
            if remaining_requests_day is not None and float(remaining_requests_day) <= 0:
                day_exhausted = True
        except ValueError:
            pass
        if day_exhausted:
            raise DailyQuotaExhausted(f"Quota journalier épuisé : {detail}")
        raise KeyDeadError(f"{resp.status_code}: {detail}")

    if resp.status_code == 400 and "organization_restricted" in detail:
        raise KeyDeadError(f"Organisation restreinte : {detail}")

    if resp.status_code in (401, 403):
        raise KeyDeadError(f"{resp.status_code}: {detail}")

    raise RuntimeError(f"{resp.status_code}: {detail}")


def parse_json_response(text):
    if not text:
        return None
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def empty_result():
    return {
        "adresse": "",
        "telephone_agence": "",
        "email_agence": "",
        "horaires": "",
        "gerant": {"valeur": "", "page": ""},
        "tel_gerant": {"valeur": "", "page": ""},
        "email_gerant": {"valeur": "", "page": ""},
    }


def merge_chunk_results(results):
    """Fusionne les résultats de plusieurs morceaux d'une même agence :
    on garde, pour chaque champ, la première valeur non vide rencontrée."""
    merged = empty_result()
    simple_fields = ("adresse", "telephone_agence", "email_agence", "horaires")
    dict_fields = ("gerant", "tel_gerant", "email_gerant")

    for result in results:
        if not result:
            continue
        for f in simple_fields:
            if not merged[f]:
                val = str(result.get(f, "") or "").strip()
                if val:
                    merged[f] = val
        for f in dict_fields:
            if not merged[f].get("valeur"):
                obj = result.get(f, {})
                if isinstance(obj, dict):
                    val = str(obj.get("valeur", "") or "").strip()
                    if val:
                        merged[f] = {"valeur": val, "page": obj.get("page", "")}
    return merged


def log_extraction(nom, result, pages):
    def fmt_simple(label, key):
        val = str(result.get(key, "")).strip()
        return f"    {label:<20}: {val if val else '(vide)'}"

    def fmt_gerant(label, key):
        obj = result.get(key, {}) if isinstance(result.get(key), dict) else {}
        val = str(obj.get("valeur", "")).strip()
        page_ref = obj.get("page", "")
        url = resolve_page_url(pages, page_ref)
        source = f" [source: {url}]" if url else ""
        return f"    {label:<20}: {val if val else '(vide)'}{source}"

    print(f"    📋 Extraction pour « {nom} » :")
    print(fmt_simple("Adresse", "adresse"))
    print(fmt_simple("Tél. agence", "telephone_agence"))
    print(fmt_simple("Email agence", "email_agence"))
    print(fmt_simple("Horaires", "horaires"))
    print(fmt_gerant("Gérant", "gerant"))
    print(fmt_gerant("Tél. gérant", "tel_gerant"))
    print(fmt_gerant("Email gérant", "email_gerant"))


RESULTS_LOCK = threading.Lock()


def save_result(nom, site_web, result, pages, url_list):
    record = {
        "nom": nom,
        "site_web": site_web,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "adresse": str(result.get("adresse", "")).strip(),
        "telephone_agence": str(result.get("telephone_agence", "")).strip(),
        "email_agence": str(result.get("email_agence", "")).strip(),
        "horaires": str(result.get("horaires", "")).strip(),
        "gerant": result.get("gerant", {}) if isinstance(result.get("gerant"), dict) else {},
        "tel_gerant": result.get("tel_gerant", {}) if isinstance(result.get("tel_gerant"), dict) else {},
        "email_gerant": result.get("email_gerant", {}) if isinstance(result.get("email_gerant"), dict) else {},
        "pages_sources": url_list,
    }
    for key in ("gerant", "tel_gerant", "email_gerant"):
        obj = record[key]
        if isinstance(obj, dict) and obj.get("page"):
            obj["url_source"] = resolve_page_url(pages, obj.get("page"))
    with RESULTS_LOCK:
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_agency_with_key(nom, pages, key_idx, api_key, limiter):
    """Traite une agence avec UNE SEULE clé dédiée (pas de rotation interne
    — c'est le thread appelant, dans main(), qui décide de requeue l'agence
    sur une autre clé si celle-ci meurt). Renvoie result_merged|None.
    Laisse remonter DailyQuotaExhausted/KeyDeadError si la clé meurt."""
    chunks = build_chunks(pages, MAX_CHUNK_TOKENS)
    chunk_total = len(chunks)
    chunk_results = []

    if chunk_total > 1:
        print(f"    ✂️ {nom} : découpée en {chunk_total} morceaux (clé #{key_idx + 1})")

    for i, chunk_items in enumerate(chunks, start=1):
        chunk_text = "\n".join(t for _, _, t in chunk_items)
        estimated = estimate_tokens(chunk_text) + PROMPT_OVERHEAD_TOKENS + COMPLETION_TOKENS_RESERVED

        limiter.wait_if_needed(estimated)

        label = f" (morceau {i}/{chunk_total})" if chunk_total > 1 else ""
        print(f"    → {nom}{label} — clé #{key_idx + 1}, ~{estimated} tokens estimés")

        text, usage = call_cerebras(nom, chunk_items, api_key, i, chunk_total)
        actual_tokens = usage.get("total_tokens", estimated) if usage else estimated
        limiter.record(actual_tokens)

        result = parse_json_response(text)
        if result:
            chunk_results.append(result)
        else:
            print(f"    ⚠️ JSON invalide sur ce morceau (texte reçu : {str(text)[:150]!r})")
            chunk_results.append(None)

    if not any(chunk_results):
        return None

    return merge_chunk_results(chunk_results)


def key_worker(key_idx, api_key, limiter, work_queue, agencies_by_nom,
                stats, stats_lock, daily_quota_flag, start_time, time_budget_seconds):
    """Boucle d'un thread dédié à UNE clé API : dépile des noms d'agence de
    la file partagée et les traite avec sa propre clé/limiter jusqu'à ce que
    la file soit vide, que le budget de temps soit atteint, ou que sa clé
    meure (auquel cas l'agence en cours est remise dans la file pour qu'un
    autre thread encore vivant la retente)."""
    while True:
        if time.time() - start_time >= time_budget_seconds:
            return
        try:
            nom = work_queue.get_nowait()
        except queue.Empty:
            return

        agency = agencies_by_nom[nom]
        pages = agency["pages"]

        try:
            result = process_agency_with_key(nom, pages, key_idx, api_key, limiter)
        except DailyQuotaExhausted as e:
            print(f"    📅 Clé #{key_idx + 1} : quota journalier Cerebras épuisé : {e}")
            with stats_lock:
                daily_quota_flag["hit"] = True
            work_queue.put(nom)  # une autre clé encore vivante la retentera
            return
        except KeyDeadError as e:
            print(f"    💀 Clé #{key_idx + 1} morte : {e}")
            work_queue.put(nom)
            return
        except Exception as e:
            print(f"    ❌ Erreur (pas liée à la clé) pour {nom} : {e}")
            with stats_lock:
                stats["failed"] += 1
            work_queue.task_done()
            continue

        if result:
            log_extraction(nom, result, pages)
            save_result(nom, agency["site_web"], result, pages, [p.get("url", "") for p in pages])
            with stats_lock:
                stats["done"].add(nom)
                stats["processed"] += 1
        else:
            print(f"    ⚠️ Aucun résultat exploitable pour « {nom} », sera retenté au prochain run")
            with stats_lock:
                stats["failed"] += 1

        work_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--force", action="store_true",
                         help="Relance même les agences déjà présentes dans extraction_results_finale.jsonl")
    parser.add_argument("--time-budget-minutes", type=int, default=320)
    parser.add_argument("--tpm-limit", type=int, default=TPM_LIMIT_DEFAULT,
                         help="Tokens/minute autorisés PAR CLÉ (vérifie ton dashboard Cerebras)")
    parser.add_argument("--rpm-limit", type=int, default=RPM_LIMIT_DEFAULT,
                         help="Requêtes/minute autorisées PAR CLÉ (vérifie ton dashboard Cerebras)")
    args = parser.parse_args()

    api_keys = load_api_keys(API_KEYS_ENV_VAR)
    limiters = [KeyRateLimiter(tpm_limit=args.tpm_limit, rpm_limit=args.rpm_limit) for _ in api_keys]
    start_time = time.time()
    time_budget_seconds = args.time_budget_minutes * 60

    agencies = load_agencies()
    agencies_by_nom = {a["nom"]: a for a in agencies}
    already_done = set() if args.force else load_existing_results()

    print(f"📦 {len(agencies)} agences dans {CACHE_PATH}")
    print(f"📝 Résultats déjà enregistrés dans {RESULTS_PATH} : {len(already_done)}")
    print(f"🔑 {len(api_keys)} clés Cerebras chargées -> jusqu'à {len(api_keys)} agences en parallèle "
          f"(limite : {args.tpm_limit} tokens/min et {args.rpm_limit} requêtes/min PAR clé)\n")

    eligible = [a["nom"] for a in agencies if a["nom"] not in already_done]
    skipped_already_done = len(agencies) - len(eligible)
    todo = eligible[: args.limit]
    print(f"▶️ {len(todo)} agence(s) à traiter ce run (sur {len(eligible)} restantes, "
          f"{skipped_already_done} déjà faites)\n")

    work_queue: queue.Queue = queue.Queue()
    for nom in todo:
        work_queue.put(nom)

    stats = {"done": set(), "processed": 0, "failed": 0}
    stats_lock = threading.Lock()
    daily_quota_flag = {"hit": False}

    threads = []
    for key_idx, api_key in enumerate(api_keys):
        t = threading.Thread(
            target=key_worker,
            args=(key_idx, api_key, limiters[key_idx], work_queue, agencies_by_nom,
                  stats, stats_lock, daily_quota_flag, start_time, time_budget_seconds),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Ce qui reste dans la file (budget temps atteint avant la fin, ou toutes
    # les clés mortes pendant que des agences y étaient encore) :
    leftover_in_queue = work_queue.qsize()
    all_done = already_done | stats["done"]
    remaining = leftover_in_queue > 0 or any(a["nom"] not in all_done for a in agencies)

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"remaining={'true' if remaining else 'false'}\n")
            fh.write(f"daily_quota_exhausted={'true' if daily_quota_flag['hit'] else 'false'}\n")

    print(f"\n✅ Terminé : {stats['processed']} agence(s) traitée(s) avec succès ce run. "
          f"{stats['failed']} échec(s) exploitable(s). {skipped_already_done} déjà faites. "
          f"Reste dans la file (non traité, budget temps) : {leftover_in_queue}. "
          f"Reste à faire au global : {remaining}")
    if daily_quota_flag["hit"]:
        print("📅 Quota journalier Cerebras atteint sur au moins une clé — "
              "le run automatique du lendemain reprendra là où on s'est arrêté "
              "(pas la peine de relancer dans les 5 minutes).")
    print(f"👉 Résultats écrits dans {RESULTS_PATH}. {CACHE_PATH} n'a PAS été modifié.")


if __name__ == "__main__":
    main()
