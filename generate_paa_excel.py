# streamlit_app.py
# -*- coding: utf-8 -*-

import os, io, json, time, math, re
from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import requests

# =============================
# Config UI & constantes
# =============================
st.set_page_config(page_title="Générateur de PAA (centré produit)", layout="wide")

DEFAULT_SHEET = "MODULES FAQs"
DEFAULT_QA_COUNT = 8
DEFAULT_ANS_MIN = 80
DEFAULT_ANS_MAX = 160
DEFAULT_ATTEMPTS = 3
REQ_TIMEOUT = (10, 60)      # (connect timeout, read timeout) secondes
JACCARD_DUP = 0.82
COSINE_DUP  = 0.86

# Modèles par défaut
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMB_MODEL  = "text-embedding-3-small"

# =============================
# Sidebar – Réglages
# =============================
st.sidebar.title("⚙️ Réglages")

chat_model = st.sidebar.text_input("Modèle Chat", os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL))
emb_model  = st.sidebar.text_input("Modèle Embeddings", os.getenv("EMB_MODEL", DEFAULT_EMB_MODEL))
ans_min    = st.sidebar.number_input("Longueur min réponse", min_value=40, max_value=300, value=DEFAULT_ANS_MIN, step=5)
ans_max    = st.sidebar.number_input("Longueur max réponse", min_value=60, max_value=400, value=DEFAULT_ANS_MAX, step=5)
max_qa     = st.sidebar.slider("Nombre de PAA par ligne", 1, 12, DEFAULT_QA_COUNT)
attempts   = st.sidebar.slider("Tentatives par ligne", 1, 5, DEFAULT_ATTEMPTS)

st.sidebar.markdown("---")
use_embeddings = st.sidebar.checkbox("Activer dédup sémantique (embeddings)", value=True)
jaccard_thr    = st.sidebar.slider("Seuil Jaccard (doublon quasi-texte)", 0.5, 0.95, JACCARD_DUP, 0.01)
cosine_thr     = st.sidebar.slider("Seuil cosinus (doublon sémantique)", 0.6, 0.95, COSINE_DUP, 0.01, disabled=not use_embeddings)
dry_run        = st.sidebar.checkbox("Dry-run (aucun appel API)", value=False)
limit_rows     = st.sidebar.number_input("Limiter n premières lignes (0 = toutes)", min_value=0, value=0, step=1)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔐 Clé API")
api_key_source = st.sidebar.radio("Où lire la clé ?", ["st.secrets", "Variable d'environnement"], index=0)
st.sidebar.caption("Ajoute OPENAI_API_KEY dans `.streamlit/secrets.toml` ou exporte la variable d’environnement.")

# Sélecteur de la source de catégorie (ton fichier n’a pas de colonne Catégorie)
cat_source = st.sidebar.radio(
    "Source de la catégorie",
    ["Colonne Catégorie", "Mots clés (fallback volontaire)"],
    index=1  # par défaut: Mots clés
)

# =============================
# Règles d’angles par catégorie (optionnel YAML)
# =============================
st.sidebar.markdown("### 📚 Règles de catégories (optionnel)")
yaml_file = st.sidebar.file_uploader("Charger un YAML de règles (facultatif)", type=["yml", "yaml"])

# Règles intégrées (fallback si pas de YAML)
DEFAULT_CATEGORY_RULES = [
    (r"(chaussette|chaussettes|socquettes|mi[-\s]?chaussette)", [
        "matières & mailles (coton bio peigné, renforts zones d’usure)",
        "respirabilité & confort au quotidien",
        "hauteurs & usages (socquettes, ville, sport doux)",
        "pointures & ajustement (entre deux tailles)",
        "entretien & durabilité (anti-boulochage sans promesse technique)",
    ]),
    (r"(boxer|slip|caleçon|shorty|culotte)", [
        "confort & maintien (coupes, ceinture douce, coutures plates)",
        "matières (coton bio, élasticité mesurée, douceur)",
        "respirabilité & usage quotidien",
        "choix de la taille et de la coupe",
        "entretien & longévité (conservation de la tenue)",
    ]),
    (r"(t[-\s]?shirt|tee[-\s]?shirt|marinière|top|haut)", [
        "coupes (droite, ajustée, unisexe) & conseils d’ajustement",
        "grammage & toucher, tenue du col",
        "superposition & usages (quotidien, layering)",
        "guide des tailles et stature",
        "entretien (lavage, séchage doux) & stabilité",
    ]),
    (r"(débardeur|brassière|caraco)", [
        "coupes (dos, bretelles) & confort",
        "matières (coton bio, maintien léger)",
        "usage (sous-pull, été, sport doux)",
        "guide des tailles (poitrine/torse)",
        "entretien & stabilité dimensionnelle",
    ]),
    (r"(thermique|thermo|grand froid|hiver|isotherme)", [
        "stratégies de superposition (base layer)",
        "matières & tricotage favorisant la chaleur ressentie",
        "respirabilité pour éviter l’humidité",
        "coupe près du corps vs confort",
        "entretien pour préserver la performance d’usage",
    ]),
    (r"(legging|collant|bas|pantalon doux)", [
        "opacité, extensibilité & confort",
        "matières (coton bio) & tenue",
        "guide des tailles (stature, hanches)",
        "entretien (lavage doux, séchage)",
        "usages (ville, homewear, mi-saison)",
    ]),
    (r"(pyjama|homewear|loungewear|peignoir)", [
        "confort & douceur (coupes, finitions)",
        "respirabilité nocturne (coton bio)",
        "choix de la taille (coupe ample/ajustée)",
        "entretien & fréquence de lavage",
        "mix & match avec d’autres pièces",
    ]),
]
DEFAULT_FALLBACK_ANGLES = [
    "matières & toucher (coton bio, fibres recyclées adaptées au produit)",
    "coupe & confort adaptés à l’usage (quotidien, layering, saison)",
    "guide des tailles (entre-deux, stature, ajustement)",
    "entretien & longévité (lavage, séchage)",
    "usages types (ville, intérieur, activité douce)",
]

def load_category_rules_from_yaml(file) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    import yaml  # import tardif
    data = yaml.safe_load(file)
    rules = []
    for r in data.get("rules", []):
        rules.append((r["match"], r["angles"]))
    fallback = data.get("fallback", DEFAULT_FALLBACK_ANGLES)
    return rules, fallback

if yaml_file is not None:
    try:
        CATEGORY_RULES, FALLBACK_ANGLES = load_category_rules_from_yaml(yaml_file)
    except Exception as e:
        st.sidebar.error(f"YAML invalide : {e}")
        CATEGORY_RULES, FALLBACK_ANGLES = DEFAULT_CATEGORY_RULES, DEFAULT_FALLBACK_ANGLES
else:
    CATEGORY_RULES, FALLBACK_ANGLES = DEFAULT_CATEGORY_RULES, DEFAULT_FALLBACK_ANGLES

# =============================
# Utilitaires
# =============================
def get_api_key() -> str:
    if dry_run:
        return ""  # pas d’API requise
    if api_key_source == "st.secrets":
        return st.secrets.get("OPENAI_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")

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

def angles_for_category(category_text: str) -> List[str]:
    text = (category_text or "").lower()
    for pat, angles in CATEGORY_RULES:
        if re.search(pat, text, flags=re.I):  # pat est une chaîne regex
            return angles
    return FALLBACK_ANGLES

# =============================
# OpenAI – wrappers (requests)
# =============================
SESSION = requests.Session()
EMB_CACHE: Dict[str, List[float]] = {}

def _post_with_retries(url: str, payload: dict, headers: dict, max_retries: int = 4) -> dict:
    wait = 0.6
    for _ in range(max_retries):
        try:
            r = SESSION.post(url, headers=headers, data=json.dumps(payload), timeout=REQ_TIMEOUT)
            if 200 <= r.status_code < 300:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(wait); wait = min(wait*2, 8.0); continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(wait); wait = min(wait*2, 8.0)
    raise RuntimeError("Échec API après retries")

def openai_chat(messages: List[Dict], model: str, temperature: float, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "n": 1}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = _post_with_retries(url, payload, headers)
    return data["choices"][0]["message"]["content"]

def get_embedding(text: str, model: str, api_key: str) -> List[float]:
    key = text.strip()
    if key in EMB_CACHE: return EMB_CACHE[key]
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": model, "input": key}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = _post_with_retries(url, payload, headers)
    vec = data["data"][0]["embedding"]
    EMB_CACHE[key] = vec
    return vec

# =============================
# Prompting
# =============================
def system_prompt(ans_min: int, ans_max: int) -> str:
    return (
        "Tu es un rédacteur SEO e-commerce. Tu crées des paires Q/R (People Also Ask) "
        "centrées sur le TYPE DE PRODUIT, pas la marque.\n"
        f"- 1 idée par question, réponses {ans_min}-{ans_max} caractères.\n"
        "- Pas de prix/promo/médical/superlatifs absolus. Français clair, vouvoiement.\n"
        '- Règle: remplacer "Achel par Lemahieu" par "Maison Lemahieu".'
    )

def user_prompt(category: str, keywords: str, angles: List[str],
                avoid_questions: List[str], need: int, ans_min: int, ans_max: int) -> str:
    obj = {
        "task": f"Générer {need} paires Q/R PAA pour une page produit.",
        "category": category, "keywords": keywords,
        "angles_prioritaires": angles,
        "constraints": {
            "answers_length": f"{ans_min}-{ans_max}",
            "one_idea_per_question": True,
            "no_prices_no_promos": True,
            "no_medical_claims": True,
            "brand_rule": 'Remplacer "Achel par Lemahieu" par "Maison Lemahieu"'
        },
        "avoid_questions": avoid_questions,
        "output_json": 'RENVOIE EXACTEMENT: {"pairs":[{"q":"...","a":"..."}]} sans texte autour.'
    }
    return json.dumps(obj, ensure_ascii=False)

def parse_pairs(text: str, ans_min: int, ans_max: int) -> List[Dict[str,str]]:
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
        if ans_min <= len(a) <= ans_max:
            out.append({"q": q, "a": a})
    return out

# =============================
# Génération d’une ligne
# =============================
def generate_for_row(category: str, keywords: str, avoid_line_qs: List[str],
                     cat_state: Dict[str, Dict[str, list]],
                     attempts: int, ans_min: int, ans_max: int,
                     max_qa: int, chat_model: str, emb_model: str,
                     use_embeddings: bool, dry_run: bool, api_key: str,
                     jaccard_thr: float, cosine_thr: float) -> List[Dict[str,str]]:

    cat_key = category or "_GLOBAL_"
    if cat_key not in cat_state: cat_state[cat_key] = {"qs": [], "embs": []}

    # Angles pilotés par la catégorie (ou mots-clés si tu l'as choisi en sidebar)
    angles = angles_for_category(category)
    avoid_norm = {normalize(q) for q in avoid_line_qs}

    for attempt in range(attempts):
        # Candidats
        if dry_run:
            candidates = [
                {"q": f"{category.strip()}: {a.split('(')[0].strip()} ?",
                 "a": (f"Réponse synthétique sur {a.split('(')[0].strip()}, "
                       f"orientée usage et entretien. Choisissez une taille adaptée.")[:ans_max]}
                for a in angles
            ][:max_qa]
        else:
            content = openai_chat(
                [{"role":"system","content":system_prompt(ans_min, ans_max)},
                 {"role":"user","content":user_prompt(category, keywords, angles, list(avoid_norm), max_qa, ans_min, ans_max)}],
                model=chat_model, temperature=0.35 if attempt == 0 else 0.6, api_key=api_key
            )
            candidates = parse_pairs(content, ans_min, ans_max)

        # Filtre Jaccard intra-ligne + historique catégorie
        filtered: List[Dict[str,str]] = []
        for p in candidates:
            q = p["q"]
            if normalize(q) in avoid_norm:
                continue
            if any(jaccard(q, x["q"]) >= jaccard_thr for x in filtered):
                continue
            if any(jaccard(q, hq) >= jaccard_thr for hq in cat_state[cat_key]["qs"]):
                continue
            filtered.append(p)
            if len(filtered) == max_qa: break

        # Filtre embeddings (sémantique)
        if use_embeddings and not dry_run and filtered:
            cand_embs = [get_embedding(p["q"], emb_model, api_key) for p in filtered]
            keep = []
            for i, ei in enumerate(cand_embs):
                ok = True
                for j in keep:
                    if cosine(ei, cand_embs[j]) >= cosine_thr:
                        ok = False; break
                if ok:
                    for eh in cat_state[cat_key]["embs"]:
                        if cosine(ei, eh) >= cosine_thr:
                            ok = False; break
                if ok: keep.append(i)
            filtered = [filtered[k] for k in keep]

        final = filtered[:max_qa]
        # MàJ historique
        cat_state[cat_key]["qs"].extend([p["q"] for p in final])
        if use_embeddings and not dry_run:
            cat_state[cat_key]["embs"].extend([get_embedding(p["q"], emb_model, api_key) for p in final])

        # suffisant ? sinon retente
        if len(final) >= max(5, max_qa - 2) or attempt == attempts - 1:
            return final
        time.sleep(0.4 if not dry_run else 0.05)

    return []

# =============================
# UI principale
# =============================
st.title("🧩 Générateur de PAA – Centré produit")

st.markdown("""
Charge un **Excel** et génère des **People Also Ask** variés, adaptés à chaque **catégorie**.
- Forte **diversité** (Jaccard + embeddings optionnels)
- Règle marque : “Achel par Lemahieu” → **“Maison Lemahieu”**
- Résultats **Q1..Q8 / A1..A8** réécrits/complétés dans un **nouvel Excel** téléchargeable.
""")

uploaded = st.file_uploader("📄 Fichier Excel d’entrée", type=["xlsx"])
if not uploaded:
    st.info("Charge ton fichier `.xlsx` pour commencer.")
    st.stop()

# Prévisualisation & choix d’onglet
xls = pd.ExcelFile(uploaded)
sheet_name = st.selectbox("Onglet à utiliser", options=xls.sheet_names, index=(xls.sheet_names.index(DEFAULT_SHEET) if DEFAULT_SHEET in xls.sheet_names else 0))
df = pd.read_excel(uploaded, sheet_name=sheet_name, engine="openpyxl")

st.write("Aperçu (5 premières lignes) :")
st.dataframe(df.head())

# Détection éventuelle d'une colonne Catégorie
cat_col = None
for cand in ["Catégorie", "Categorie", "Catégorie produit", "Categorie produit"]:
    if cand in df.columns:
        cat_col = cand
        break

col1, col2 = st.columns(2)
with col1:
    if cat_source == "Colonne Catégorie":
        if cat_col:
            st.success(f"Colonne de Catégorie détectée : **{cat_col}**")
        else:
            st.error("Aucune colonne de **Catégorie** détectée. "
                     "Ajoutez une colonne `Catégorie`/`Categorie` ou sélectionnez "
                     "**Mots clés (fallback volontaire)** dans la barre latérale.")
            st.stop()
    else:
        st.info("Catégorie = **Mots clés** (fallback volontaire).")
with col2:
    st.write("Colonnes minimales : **Adresse | Mots clés**")

# Validation colonnes minimales (Priorité devient optionnelle)
required_cols = ["Adresse", "Mots clés"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes : {', '.join(missing)}")
    st.stop()

# Ajouter colonnes Q/A si absentes
q_cols = [f"Q{i}" for i in range(1, max_qa+1)]
a_cols = [f"A{i}" for i in range(1, max_qa+1)]
for qc, ac in zip(q_cols, a_cols):
    if qc not in df.columns: df[qc] = ""
    if ac not in df.columns: df[ac] = ""

# Bouton lancer
start = st.button("🚀 Générer les PAA")
if not start:
    st.stop()

# Vérif clé API si nécessaire
api_key = get_api_key()
if not dry_run and not api_key:
    st.error("Aucune clé API détectée. Ajoute `OPENAI_API_KEY` dans `st.secrets` ou en variable d’environnement, ou coche Dry-run.")
    st.stop()

# Pipeline
out = df.copy()
total = len(df) if limit_rows == 0 else min(limit_rows, len(df))
progress = st.progress(0, text="Démarrage…")
log_area = st.empty()

cat_state: Dict[str, Dict[str, list]] = {}
for idx in range(total):
    row = df.iloc[idx]
    keywords = str(row.get("Mots clés", "") or "")

    if cat_source == "Mots clés (fallback volontaire)":
        category = keywords
    else:
        category = str(row.get(cat_col, "") or "")
        if not category:
            st.warning(f"Ligne {idx+2} : catégorie vide — ligne ignorée.")
            continue

    # Éviter de répéter exact les Q déjà présentes (si tu as un historique dans le fichier)
    avoid_line_qs = []
    for qc in q_cols:
        v = str(row.get(qc, "") or "").strip()
        if v: avoid_line_qs.append(v)

    try:
        pairs = generate_for_row(
            category=category, keywords=keywords, avoid_line_qs=avoid_line_qs,
            cat_state=cat_state, attempts=attempts,
            ans_min=ans_min, ans_max=ans_max, max_qa=max_qa,
            chat_model=chat_model, emb_model=emb_model,
            use_embeddings=use_embeddings, dry_run=dry_run, api_key=api_key,
            jaccard_thr=jaccard_thr, cosine_thr=cosine_thr
        )
    except Exception as e:
        st.error(f"Erreur à la ligne {idx+2} : {e}")
        pairs = []

    for i in range(max_qa):
        out.at[idx, q_cols[i]] = pairs[i]["q"] if i < len(pairs) else ""
        out.at[idx, a_cols[i]] = pairs[i]["a"] if i < len(pairs) else ""

    progress.progress((idx+1)/total, text=f"Lignes traitées : {idx+1}/{total}")
    if (idx+1) % 5 == 0 or idx == total-1:
        log_area.info(f"[{idx+1}/{total}] Catégorie: {category[:60]} — Q générées: {len(pairs)}")

progress.empty()
st.success("✅ Terminé !")

# Export Excel en mémoire + bouton de téléchargement
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name=sheet_name)
buf.seek(0)
st.download_button(
    label="💾 Télécharger l’Excel de sortie",
    data=buf,
    file_name="output_paa.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Astuce : utilise **Dry-run** pour valider le flux sans consommer l’API. Active ensuite les embeddings pour maximiser la diversité.")
