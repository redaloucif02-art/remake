"""
Fusionne le cache retraité (les 213 agences relancées avec le scraper corrigé)
dans le cache original du run complet sur les 6000 agences.

Usage :
    python3 fusionner_caches.py pages_cache_finale.jsonl "pages_cache_finale (1).jsonl"

Clé de dédoublonnage : site_web (plus fiable que le nom, qui peut varier en
casse/espaces). En cas de doublon entre les deux fichiers, la version issue
du fichier RETRAITÉ (deuxième argument) l'emporte -- elle est plus récente
et vient du scraper corrigé.

Écrit pages_cache_finale_merged.jsonl (ne touche pas les fichiers d'origine,
au cas où il faudrait comparer ou refaire).
"""
import json
import sys

if len(sys.argv) != 3:
    print("Usage : python3 fusionner_caches.py <original.jsonl> <retraite.jsonl>")
    sys.exit(1)

original_path, retraite_path = sys.argv[1], sys.argv[2]
output_path = "pages_cache_finale_merged.jsonl"


def load(path: str) -> dict[str, dict]:
    """Charge un cache en dict {site_web: entrée}. Ignore les lignes
    vides/mal formées plutôt que de planter dessus."""
    entries = {}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            key = entry.get("site_web", "").strip()
            if not key:
                skipped += 1
                continue
            entries[key] = entry
    if skipped:
        print(f"  ({skipped} ligne(s) ignorée(s) dans {path} : vide ou mal formée)")
    return entries


original = load(original_path)
retraite = load(retraite_path)

print(f"Original  : {len(original)} agence(s)")
print(f"Retraité  : {len(retraite)} agence(s)")

overlap = set(original) & set(retraite)
if overlap:
    print(f"  {len(overlap)} agence(s) présente(s) dans les deux -> version retraitée gardée")

merged = {**original, **retraite}  # retraite écrase original en cas de clé commune

with open(output_path, "w", encoding="utf-8") as f:
    for entry in merged.values():
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nFusion terminée : {len(merged)} agence(s) -> {output_path}")
print(f"  (attendu : {len(original)} + {len(retraite)} - {len(overlap)} = {len(original) + len(retraite) - len(overlap)})")
