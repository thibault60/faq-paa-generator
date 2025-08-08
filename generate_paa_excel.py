# generate_paa_excel.py
# -*- coding: utf-8 -*-

import os, json, time, math, re, argparse, logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import requests

# ====== Config ======
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMB_MODEL  = os.getenv("EMB_MODEL",  "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_QA      = 8
ANS_MIN     = 80
ANS_MAX     = 160
ATTEMPTS    = 3              # tentatives de génération par ligne
REQ_TIMEOUT = (10, 60)       # (connect timeout, read timeout)
JACCARD_DUP = 0.82
COSINE_DUP  = 0.86

SESSION = requests.Session()

def api_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def _post(url: str, payload: dict, headers: dict, max_retries: int = 4) -> dict:
    wait = 0.6
    for attempt in range(max_retries):
        try:
            r = SESSION.post(url, headers=headers, data=json.dumps(payload), timeout=REQ_TIMEOUT)
            if 200 <= r.status_code < 300:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(wait); wait = min(wait*2, 8.0); continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            if attempt == max_retries-1: raise
            time.sleep(wait); wait = min(wait*2, 8.0)
    raise RuntimeError("Échec API après retries")

def openai_chat(messages: List[Dict], temperature: float) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant.")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": temperature, "n": 1}
    data = _post(url, payload, api_headers())
    return data["choices"][0]["message"]["content"]

EMB_CACHE: Dict[str, List[float]] = {}
def embedding(text: str) -> List[float]:
    key = text.strip()
    if key in EMB_CACHE: return EMB_CACHE[key]
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": EMB_MODEL, "input": key}
    data = _post(url, payload, api_headers())
    vec = data["data"][0]["embedding"]
    EMB_CACHE[key] = vec
    return vec

# ====== Similarité / diversité ======
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"achel\s+par\s+lemahieu", "maison lemahieu", s, flags=re.I)
    s = re.sub(r"[^0-9a-zàâçéèêëîïôûùüÿñœæ\s-]+", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard(a: str, b: str) -> float:
    A, B = set(normalize(a).split()), set(normalize(b).split())
    if not A and not B: return 1.0
    inter, uni = len(A & B), len(A | B)
    return inter/uni if uni else 0.0

def cosine(u: List[float], v: List[float]) -> float:
    if not u or not v: return 0.0
    num = sum(x*y for x,y in zip(u,v))
    du  = math.sqrt(sum(x*x for x in u))
    dv  = math.sqrt(sum(y*y for y in v))
    return 0.0 if du == 0 or dv == 0 else num/(du*dv)

# ====== Angles par catégorie (centré PRODUIT) ======
CATEGORY_RULES: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(r"(chaussette|chaussettes|socquettes|mi[-\s]?chaussette)", re.I),
     ["matières & mailles (coton bio peigné, renforts)",
      "respirabilité & confort", "hauteurs & usages (socquettes, ville)",
      "pointures & ajustement", "entretien & durabilité"]),
    (re.compile(r"(boxer|slip|caleçon|shorty|culotte)", re.I),
     ["confort & maintien (coupes, ceinture douce)",
      "matières (coton bio, élasticité)", "respirabilité au quotidien",
      "guide des tailles/coupe", "entretien & tenue"]),
    (re.compile(r"(t[-\s]?shirt|tee[-\s]?shirt|marinière|top|haut)", re.I),
     ["coupes (droite, ajustée, unisexe)", "grammage & toucher, tenue du col",
      "superposition & usages", "guide des tailles", "entretien & stabilité"]),
    (re.compile(r"(débardeur|brassière|caraco)", re.I),
     ["coupes (dos, bretelles) & confort", "matières (coton bio, maintien léger)",
      "usages (été, sous-pull, sport doux)", "guide des tailles", "entretien & stabilité"]),
    (re.compile(r"(thermique|thermo|grand froid|hiver|isotherme)", re.I),
     ["base layer & superposition", "matières favorisant la chaleur ressentie",
      "respirabilité (éviter l’humidité)", "coupe près du corps vs confort",
      "entretien pour préserver l’usage"]),
    (re.compile(r"(legging|collant|bas|pantalon doux)", re.I),
     ["opacité & extensibilité", "matières (coton bio) & tenue",
      "guide des tailles (stature, hanches)", "entretien (lavage doux)",
      "usages (ville, homewear)"]),
    (re.compile(r"(pyjama|homewear|loungewear|peignoir)", re.I),
     ["confort & douceur", "respirabilité nocturne",
      "choix de la taille (ample/ajusté)", "entretien & fréquence", "mix & match"]),
]
FALLBACK_ANGLES = [
    "matières & toucher (coton bio, fibres recyclées)",
    "coupe & confort adaptés à l’usage",
    "guide des tailles (entre-deux, stature)",
    "entretien & longévité (lavage, séchage)",
    "usages types (quotidien, layering, saison)"
]

def angles_for_category(category_or_kw: str) -> List[str]:
    text = (category_or_kw or "").lower()
    for pat, angles in CATEGORY_RULES:
        if pat.search(text): return angles
    return FALLBACK_ANGLES

# ====== Prompting ======
def system_prompt() -> str:
    return (
        "Tu es un rédacteur SEO e-commerce. Crée des paires Q/R (People Also Ask) "
        "centrées sur le TYPE DE PRODUIT, pas la marque.\n"
        f"- 1 idée par question, réponses {ANS_MIN}-{ANS_MAX} caractères.\n"
        "- Pas de prix/promo/médical/superlatifs absolus. Français clair, vouvoiement.\n"
        '- Règle: remplacer "Achel par Lemahieu" par "Maison Lemahieu".'
    )

def user_prompt(category: str, keywords: str, angles: List[str],
                avoid_questions: List[str], need: int) -> str:
    obj = {
        "task": f"Générer {need} paires Q/R PAA pour une page produit.",
        "category": category, "keywords": keywords,
        "angles_prioritaires": angles,
        "constraints": {
            "answers_length": f"{ANS_MIN}-{ANS_MAX}",
            "one_idea_per_question": True,
            "no_prices_no_promos": True,
            "no_medical_claims": True,
            "brand_rule": 'Remplacer "Achel par Lemahieu" par "Maison Lemahieu"'
        },
        "avoid_questions": avoid_questions,
        "output_json": 'RENVOIE EXACTEMENT: {"pairs":[{"q":"...","a":"..."}]} sans texte autour.'
    }
    return json.dumps(obj, ensure_ascii=False)

def parse_pairs(text: str) -> List[Dict[str, str]]:
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m: return []
        try: data = json.loads(m.group(0))
        except Exception: return []
    out = []
    for p in data.get("pairs", []):
        q = str(p.get("q","")).strip()
        a = str(p.get("a","")).strip()
        if not q or not a: continue
        q = re.sub(r"achel\s+par\s+lemahieu", "Maison Lemahieu", q, flags=re.I)
        a = re.sub(r"achel\s+par\s+lemahieu", "Maison Lemahieu", a, flags=re.I)
        if ANS_MIN <= len(a) <= ANS_MAX:
            out.append({"q": q, "a": a})
    return out

# ====== Génération & diversité ======
def generate_for_row(category: str, keywords: str, avoid_line_qs: List[str],
                     cat_state: Dict[str, Dict[str, list]],
                     attempts: int, dry_run: bool, verbose: bool) -> List[Dict[str,str]]:
    cat_key = category or keywords or "_GLOBAL_"
    if cat_key not in cat_state: cat_state[cat_key] = {"qs": [], "embs": []}

    angles = angles_for_category(category or keywords)
    avoid_norm = {normalize(q) for q in avoid_line_qs}

    for attempt in range(attempts):
        # Génération candidates
        if dry_run:
            candidates = [
                {"q": f"{(category or keywords).strip()}: {a.split('(')[0].strip()} ?",
                 "a": (f"Réponse synthétique sur {a.split('(')[0].strip()}, "
                       f"orientée usage et entretien. Choisissez une taille adaptée.")[:ANS_MAX]}
                for a in angles
            ][:MAX_QA]
        else:
            content = openai_chat(
                [{"role":"system","content":system_prompt()},
                 {"role":"user","content":user_prompt(category, keywords, angles, list(avoid_norm), MAX_QA)}],
                temperature=0.35 if attempt == 0 else 0.6
            )
            candidates = parse_pairs(content)

        # 1) Filtre Jaccard intra-ligne + vs historique catégorie
        filtered: List[Dict[str,str]] = []
        for p in candidates:
            q = p["q"]
            if normalize(q) in avoid_norm:       # évite exactes/voisines à la ligne
                continue
            if any(jaccard(q, x["q"]) >= JACCARD_DUP for x in filtered):
                continue
            if any(jaccard(q, hq) >= JACCARD_DUP for hq in cat_state[cat_key]["qs"]):
                continue
            filtered.append(p)
            if len(filtered) == MAX_QA: break

        # 2) Filtre embeddings (sémantique) intra-ligne + historique
        if not dry_run and filtered:
            cand_embs = [embedding(p["q"]) for p in filtered]
            keep = []
            for i, ei in enumerate(cand_embs):
                ok = True
                for j in keep:
                    if cosine(ei, cand_embs[j]) >= COSINE_DUP:
                        ok = False; break
                if ok:
                    for eh in cat_state[cat_key]["embs"]:
                        if cosine(ei, eh) >= COSINE_DUP:
                            ok = False; break
                if ok: keep.append(i)
            filtered = [filtered[k] for k in keep]

        if verbose: logging.info(f"[{cat_key}] tentative {attempt+1}: {len(filtered)} retenues")
        final = filtered[:MAX_QA]

        # mise à jour historique
        cat_state[cat_key]["qs"].extend([p["q"] for p in final])
        if not dry_run:
            cat_state[cat_key]["embs"].extend([embedding(p["q"]) for p in final])

        if len(final) >= max(5, MAX_QA-2) or attempt == attempts-1:
            return final
        time.sleep(0.5 if not dry_run else 0.05)

    return []

# ====== Excel ======
def load_excel(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

def save_excel(df: pd.DataFrame, path: str):
    df.to_excel(path, index=False, engine="openpyxl")

# ====== Main ======
def main():
    ap = argparse.ArgumentParser(description="Génération PAA (Excel in/out), centrée PRODUIT, forte diversité.")
    ap.add_argument("--input", required=True, help="Chemin Excel source")
    ap.add_argument("--sheet", default="MODULES FAQs", help="Nom d'onglet source (par défaut: MODULES FAQs)")
    ap.add_argument("--output", default="output_paa.xlsx", help="Excel de sortie")
    ap.add_argument("--max-rows", type=int, default=None, help="Limiter le nb de lignes (debug)")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS, help="Tentatives par ligne")
    ap.add_argument("--dry-run", action="store_true", help="Aucun appel API (test E/S)")
    ap.add_argument("--verbose", action="store_true", help="Logs détaillés")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.dry_run and not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant. export OPENAI_API_KEY=...")

    df = load_excel(args.input, args.sheet)

    # Vérifs colonnes
    for col in ["Adresse", "Mots clés", "Priorité"]:
        if col not in df.columns:
            raise ValueError(f"Colonne obligatoire manquante: {col}")

    out = df.copy()

    # Colonnes Q/A garanties
    q_cols = [f"Q{i}" for i in range(1, MAX_QA+1)]
    a_cols = [f"A{i}" for i in range(1, MAX_QA+1)]
    for qc, ac in zip(q_cols, a_cols):
        if qc not in out.columns: out[qc] = ""
        if ac not in out.columns: out[ac] = ""

    # repère colonne catégorie optionnelle
    cat_col = None
    for cand in ["Catégorie", "Categorie", "Catégorie produit", "Categorie produit"]:
        if cand in df.columns: cat_col = cand; break

    total = len(df) if args.max_rows is None else min(args.max_rows, len(df))
    pbar = tqdm(total=total, desc="Génération PAA", unit="ligne")
    cat_state: Dict[str, Dict[str, list]] = {}

    for idx in range(total):
        row = df.iloc[idx]
        keywords = str(row.get("Mots clés", "") or "")
        category = str(row.get(cat_col, "") or "") if cat_col else keywords

        # éviter les questions déjà présentes (si ton fichier en contient)
        avoid_line_qs = []
        for qc in q_cols:
            v = str(row.get(qc, "") or "").strip()
            if v: avoid_line_qs.append(v)

        pairs = generate_for_row(
            category, keywords, avoid_line_qs,
            cat_state=cat_state, attempts=args.attempts,
            dry_run=args.dry_run, verbose=args.verbose
        )

        for i in range(MAX_QA):
            out.at[idx, q_cols[i]] = pairs[i]["q"] if i < len(pairs) else ""
            out.at[idx, a_cols[i]] = pairs[i]["a"] if i < len(pairs) else ""

        pbar.update(1)

    pbar.close()
    save_excel(out, args.output)
    print(f"✅ Terminé. Fichier écrit : {args.output}")

if __name__ == "__main__":
    main()
