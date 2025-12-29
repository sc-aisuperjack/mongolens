import os
import io
import re
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util

from openai import OpenAI


APP_TITLE = "MongoLens — Mongo → Insights → RAG Chat"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-5"  # change if your org doesn't have access

ENV_MAP = {
    "local": ".env.local",
    "dev": ".env.dev",
    "test": ".env.test",
    "live": ".env.live",
}

# ---------- Helpers ----------

def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets if present, fallback to env var
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

def load_env_file(env_key: str) -> None:
    env_file = ENV_MAP.get(env_key, ".env.local")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

def bson_to_python(obj: Any) -> Any:
    # Convert BSON types safely to JSON-friendly Python types
    return json.loads(json_util.dumps(obj))

def safe_str(v: Any, max_len: int = 400) -> str:
    s = str(v)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] + ("…" if len(s) > max_len else "")

def record_to_text(collection: str, rec: Dict[str, Any], max_chars: int = 1800) -> List[Tuple[str, str]]:
    """
    Turn a Mongo record into one or more text chunks.
    Returns list of (chunk_id, chunk_text)
    """
    rid = safe_str(rec.get("_id", "no_id"), 80)
    base = [f"collection: {collection}", f"_id: {rid}"]

    # Flatten-ish: key: value lines (truncate big values)
    lines = []
    for k, v in rec.items():
        if k == "_id":
            continue
        if isinstance(v, (dict, list)):
            val = safe_str(json.dumps(v, ensure_ascii=False), 500)
        else:
            val = safe_str(v, 500)
        lines.append(f"{k}: {val}")

    full = "\n".join(base + lines)

    # If too long, chunk it
    if len(full) <= max_chars:
        return [(f"{collection}:{rid}:0", full)]

    chunks = []
    parts = full.split("\n")
    cur = []
    cur_len = 0
    idx = 0
    for line in parts:
        if cur_len + len(line) + 1 > max_chars and cur:
            chunks.append((f"{collection}:{rid}:{idx}", "\n".join(cur)))
            idx += 1
            cur = []
            cur_len = 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append((f"{collection}:{rid}:{idx}", "\n".join(cur)))
    return chunks

def cosine_top_k(matrix: np.ndarray, q: np.ndarray, k: int = 5) -> List[int]:
    # matrix: [n, d], q: [d]
    q = q / (np.linalg.norm(q) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return idx.tolist()

@st.cache_data(show_spinner=False)
def embed_texts(api_key: str, model: str, texts: List[str]) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    # OpenAI embeddings endpoint supports batching inputs. :contentReference[oaicite:3]{index=3}
    out: List[List[float]] = []
    batch = 128
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i+batch])
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float32)

def call_grounded_chat(api_key: str, model: str, question: str, context_chunks: List[Tuple[str, str]]) -> str:
    client = OpenAI(api_key=api_key)

    context_block = "\n\n".join([f"[{cid}]\n{ctext}" for cid, ctext in context_chunks])

    developer_msg = (
        "You are a data analyst assistant. You must answer ONLY using the provided CONTEXT.\n"
        "If the answer is not clearly supported by the context, say: \"Not found in the supplied records.\".\n"
        "Always include citations using the chunk ids in square brackets, for example: [collection:id:0].\n"
        "Do not guess or invent fields, numbers, or relationships."
    )

    user_msg = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Return a concise answer with citations."
    )

    # Responses API is the recommended API for new builds. :contentReference[oaicite:4]{index=4}
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": developer_msg},
            {"role": "user", "content": user_msg},
        ],
        
    )
    return resp.output_text  # shown in OpenAI quickstart examples :contentReference[oaicite:5]{index=5}

# ---------- Insights / chart suggestions ----------

@dataclass
class ChartSuggestion:
    title: str
    kind: str
    x: str
    y: Optional[str] = None

def infer_chart_suggestions(df: pd.DataFrame) -> List[ChartSuggestion]:
    suggestions: List[ChartSuggestion] = []

    if df.empty:
        return suggestions

    # Date-like columns
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]

    # If rating exists, show distribution
    for c in df.columns:
        if c.lower() in ["rating", "stars", "score"]:
            if pd.api.types.is_numeric_dtype(df[c]):
                suggestions.append(ChartSuggestion(title="Rating distribution", kind="hist", x=c))

    # If have date + numeric, line chart over time
    if date_cols and num_cols:
        suggestions.append(ChartSuggestion(title="Trend over time", kind="line", x=date_cols[0], y=num_cols[0]))

    # If categorical, bar top values
    if cat_cols:
        suggestions.append(ChartSuggestion(title="Top categories", kind="bar", x=cat_cols[0], y="count"))

    # If 2 numeric columns, scatter
    if len(num_cols) >= 2:
        suggestions.append(ChartSuggestion(title="Numeric relationship", kind="scatter", x=num_cols[0], y=num_cols[1]))

    return suggestions[:4]

def render_suggestion(df: pd.DataFrame, s: ChartSuggestion) -> None:
    import altair as alt

    if s.kind == "hist":
        chart = alt.Chart(df.dropna(subset=[s.x])).mark_bar().encode(
            x=alt.X(s.x, bin=True),
            y="count()",
        )
        st.altair_chart(chart, use_container_width=True)

    elif s.kind == "line" and s.y:
        d = df.copy()
        d[s.x] = pd.to_datetime(d[s.x], errors="coerce")
        d = d.dropna(subset=[s.x, s.y]).sort_values(s.x)
        chart = alt.Chart(d).mark_line().encode(
            x=s.x,
            y=s.y,
        )
        st.altair_chart(chart, use_container_width=True)

    elif s.kind == "bar":
        top = df[s.x].astype(str).value_counts().head(15).reset_index()
        top.columns = [s.x, "count"]
        chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("count:Q"),
            y=alt.Y(f"{s.x}:N", sort="-x"),
        )
        st.altair_chart(chart, use_container_width=True)

    elif s.kind == "scatter" and s.y:
        d = df.dropna(subset=[s.x, s.y])
        chart = alt.Chart(d).mark_point().encode(
            x=s.x,
            y=s.y,
            tooltip=[s.x, s.y],
        )
        st.altair_chart(chart, use_container_width=True)

# ---------- Streamlit UI ----------

# st.set_page_config(page_title=APP_TITLE, layout="wide")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="assets/mongolens.ico",  
    layout="wide"
)

# Display the logo and title
col1, col2 = st.columns([1, 8])
with col1:
    st.image("assets/mongolens.png", width=70)
with col2:
    st.markdown(
        "<h1 style='margin-bottom: 0; color: #4B8BBE;'>" + APP_TITLE + "</h1>"
        "<p style='margin-top: 0;'>Select collections, export JSON, generate grounded insights, and chat with RAG over your exported data.</p>",
        unsafe_allow_html=True
    )


# st.title(APP_TITLE)
# st.caption("Select collections, export JSON, generate grounded insights, and chat with RAG over your exported data.")

api_key = get_api_key()
if not api_key:
    st.warning("Set OPENAI_API_KEY in your environment or Streamlit secrets before using embeddings/chat.")

with st.sidebar:
    st.header("Connection")
    env_key = st.selectbox("Environment", list(ENV_MAP.keys()), index=0)
    load_env_file(env_key)

    mongo_uri = st.text_input("MONGO_URI", value=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db_name = st.text_input("Database name", value=os.getenv("DB_NAME", ""))

    mode = st.radio("Data source", ["Connect to Mongo", "Upload Export (zip/json)"], index=0)

tabs = st.tabs(["1) Load data", "2) Insights", "3) RAG chat"])

# Session state
st.session_state.setdefault("docs_by_coll", {})      # collection -> list[dict]
st.session_state.setdefault("df_by_coll", {})        # collection -> dataframe
st.session_state.setdefault("chunks", [])            # list[(chunk_id, chunk_text)]
st.session_state.setdefault("chunk_ids", [])         # list[str]
st.session_state.setdefault("chunk_texts", [])       # list[str]
st.session_state.setdefault("emb_matrix", None)      # np.ndarray

def rebuild_dataframes() -> None:
    df_by = {}
    for coll, docs in st.session_state["docs_by_coll"].items():
        # Normalise to tabular where possible
        try:
            df_by[coll] = pd.json_normalize(docs, sep=".")
        except Exception:
            df_by[coll] = pd.DataFrame(docs)
    st.session_state["df_by_coll"] = df_by

def build_rag_index(selected: List[str]) -> None:
    chunks: List[Tuple[str, str]] = []
    for coll in selected:
        for rec in st.session_state["docs_by_coll"].get(coll, []):
            for cid, ctext in record_to_text(coll, rec):
                chunks.append((cid, ctext))

    st.session_state["chunks"] = chunks
    st.session_state["chunk_ids"] = [c[0] for c in chunks]
    st.session_state["chunk_texts"] = [c[1] for c in chunks]

    if not api_key:
        st.error("Missing OPENAI_API_KEY. Cannot build embeddings index.")
        return

    with st.spinner("Embedding chunks..."):
        emb = embed_texts(api_key, DEFAULT_EMBED_MODEL, st.session_state["chunk_texts"])
    st.session_state["emb_matrix"] = emb

with tabs[0]:
    st.subheader("Load and select data")

    if mode == "Connect to Mongo":
        col1, col2 = st.columns([1, 1])
        with col1:
            sample_limit = st.slider("Sample docs per collection", 50, 5000, 300, step=50)
        with col2:
            export_zip_name = st.text_input("Export zip name", value="mongo_export.zip")

        if st.button("Connect and list collections", type="primary"):
            try:
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
                db = client[db_name] if db_name else None
                if not db_name:
                    st.error("Enter a database name.")
                else:
                    collections = db.list_collection_names()
                    info = []
                    for c in collections:
                        try:
                            info.append((c, db[c].estimated_document_count()))
                        except Exception:
                            info.append((c, None))
                    st.session_state["collections_info"] = sorted(info, key=lambda x: (x[1] is None, -(x[1] or 0)))
                    st.success(f"Found {len(collections)} collections.")
            except Exception as e:
                st.error(f"Mongo connection failed: {e}")

        collections_info = st.session_state.get("collections_info", [])
        if collections_info:
            st.write("Collections")
            dfc = pd.DataFrame(collections_info, columns=["collection", "estimated_docs"])
            st.dataframe(dfc, use_container_width=True, height=240)

            selected = st.multiselect(
                "Select collections to load",
                options=[c for c, _ in collections_info],
                default=[collections_info[0][0]] if collections_info else [],
            )

            if st.button("Load selected collections"):
                if not selected:
                    st.warning("Select at least one collection.")
                else:
                    try:
                        client = MongoClient(mongo_uri)
                        db = client[db_name]
                        docs_by = {}
                        for c in selected:
                            docs = list(db[c].find({}).limit(int(sample_limit)))
                            docs_by[c] = [bson_to_python(d) for d in docs]
                        st.session_state["docs_by_coll"] = docs_by
                        rebuild_dataframes()
                        st.success(f"Loaded {len(selected)} collection(s).")
                    except Exception as e:
                        st.error(f"Failed to load docs: {e}")

            if st.session_state["docs_by_coll"]:
                # Export zip
                if st.button("Export selected as zip"):
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for coll, docs in st.session_state["docs_by_coll"].items():
                            zf.writestr(f"{coll}.json", json.dumps(docs, ensure_ascii=False, indent=2))
                    mem.seek(0)
                    st.download_button(
                        "Download export",
                        data=mem,
                        file_name=export_zip_name,
                        mime="application/zip",
                    )

    else:
        st.info("Upload an export zip (collection.json files) or a single json file with a list of records.")
        up = st.file_uploader("Upload .zip or .json", type=["zip", "json"])
        if up is not None:
            docs_by = {}
            if up.name.endswith(".zip"):
                with zipfile.ZipFile(up) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".json"):
                            raw = zf.read(name).decode("utf-8", errors="ignore")
                            docs = json.loads(raw)
                            coll = os.path.splitext(os.path.basename(name))[0]
                            if isinstance(docs, list):
                                docs_by[coll] = docs
            else:
                raw = up.read().decode("utf-8", errors="ignore")
                docs = json.loads(raw)
                docs_by["uploaded"] = docs if isinstance(docs, list) else [docs]

            st.session_state["docs_by_coll"] = docs_by
            rebuild_dataframes()
            st.success(f"Loaded {len(docs_by)} collection(s) from upload.")

    # Preview
    if st.session_state["df_by_coll"]:
        st.markdown("### Preview")
        for coll, df in st.session_state["df_by_coll"].items():
            st.write(f"**{coll}** — {len(df):,} rows, {len(df.columns):,} columns")
            st.dataframe(df.head(50), use_container_width=True)

with tabs[1]:
    st.subheader("Auto insights + chart suggestions (Python-grounded)")

    if not st.session_state["df_by_coll"]:
        st.info("Load data first.")
    else:
        coll = st.selectbox("Collection", list(st.session_state["df_by_coll"].keys()))
        df = st.session_state["df_by_coll"][coll]

        st.write("Basic profiling")
        st.write({
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "null_cells": int(df.isna().sum().sum()),
        })

        # Show top columns with nulls
        nulls = df.isna().mean().sort_values(ascending=False).head(10)
        st.write("Top null-rate columns")
        st.dataframe(nulls.reset_index().rename(columns={"index": "column", 0: "null_rate"}), use_container_width=True)

        st.markdown("### Suggested charts")
        suggestions = infer_chart_suggestions(df)
        if not suggestions:
            st.info("No chart suggestions for this dataset.")
        else:
            for s in suggestions:
                st.write(f"**{s.title}** ({s.kind})")
                render_suggestion(df, s)

with tabs[2]:
    st.subheader("RAG chat grounded in your selected records")

    if not st.session_state["docs_by_coll"]:
        st.info("Load data first.")
    else:
        selected = st.multiselect(
            "Collections to index",
            options=list(st.session_state["docs_by_coll"].keys()),
            default=list(st.session_state["docs_by_coll"].keys()),
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
        with col2:
            top_k = st.slider("Top K chunks", 3, 12, 6)

        if st.button("Build / rebuild RAG index", type="primary"):
            if not selected:
                st.warning("Select at least one collection.")
            else:
                build_rag_index(selected)
                if st.session_state["emb_matrix"] is not None:
                    st.success(f"Indexed {len(st.session_state['chunks'])} chunks using {DEFAULT_EMBED_MODEL}.")

        if st.session_state["emb_matrix"] is None:
            st.info("Build the index to enable chat.")
        else:
            q = st.text_input("Ask a question about the indexed data")
            if st.button("Ask"):
                if not q.strip():
                    st.warning("Enter a question.")
                elif not api_key:
                    st.error("Missing OPENAI_API_KEY.")
                else:
                    with st.spinner("Retrieving and answering..."):
                        q_emb = embed_texts(api_key, DEFAULT_EMBED_MODEL, [q])[0]
                        idxs = cosine_top_k(st.session_state["emb_matrix"], q_emb, k=int(top_k))
                        ctx = [st.session_state["chunks"][i] for i in idxs]
                        answer = call_grounded_chat(api_key, chat_model, q, ctx)
                    st.markdown(answer)

                    with st.expander("Retrieved context chunks"):
                        for cid, ctext in ctx:
                            st.write(cid)
                            st.code(ctext)
