
import argparse, os, ast, json, sys
import numpy as np
import pandas as pd

# ------------------------ Embedders ------------------------

def _load_openai_key():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        key = cfg.get("openaiKey") or cfg.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("openaiKey missing in config.json")
        return key
    except FileNotFoundError:
        raise FileNotFoundError("config.json not found. Create it with {'openaiKey': '...'}")

def encode_query_free(q: str) -> np.ndarray:
    """Encode with SentenceTransformers (free, local)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("ERROR: sentence-transformers not installed. Try: pip install sentence-transformers", file=sys.stderr)
        raise
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode([q], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)

def encode_query_openai(q: str) -> np.ndarray:
    """Encode with OpenAI embeddings API."""
    try:
        from openai import OpenAI
    except Exception as e:
        print("ERROR: openai package not installed. Try: pip install openai", file=sys.stderr)
        raise
    client = OpenAI(api_key=_load_openai_key())
    q = q.replace("\n", " ")
    emb = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
    return np.asarray(emb, dtype=np.float32)

def pick_encoder(embedder: str):
    return encode_query_free if embedder == "free" else encode_query_openai

# ------------------------ Utilities ------------------------

def cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarities between query_vec and each row of matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q

def _parse_embedding_cell(cell: str) -> np.ndarray:
    """Embeddings are saved as stringified Python lists in the CSVs -> convert to np.array."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=np.float32)
    try:
        return np.asarray(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        # try JSON
        try:
            return np.asarray(json.loads(cell), dtype=np.float32)
        except Exception:
            raise ValueError("Could not parse embedding cell; unexpected format.")

def load_table(embedder: str, granularity: str) -> pd.DataFrame:
    base = "free" if embedder == "free" else "openai"
    if granularity == "talks":
        path = os.path.join(base, f"{base}_talks.csv")
    elif granularity == "paragraphs":
        path = os.path.join(base, f"{base}_paragraphs.csv")
    elif granularity == "clusters":
        path = os.path.join(base, f"{base}_3_clusters.csv")
    else:
        raise ValueError("granularity must be one of: talks | paragraphs | clusters")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}. Did you run the pipeline scripts?")
    df = pd.read_csv(path)
    if "embedding" not in df.columns:
        raise ValueError(f"'embedding' column not found in {path}.")
    df["embedding"] = df["embedding"].apply(_parse_embedding_cell)
    return df

def _safe_first_text(row, granularity: str) -> str:
    if granularity == "talks":
        return ""
    text = row.get("text", "")
    if isinstance(text, str):
        # clusters save top-3 paragraphs as a list-string; paragraphs save the raw string.
        try:
            maybe_list = ast.literal_eval(text)
            if isinstance(maybe_list, list):
                text = " ".join(maybe_list[:1])
        except Exception:
            pass
    return (text or "").strip().replace("\n", " ")

def format_row(row: pd.Series, granularity: str) -> str:
    who = " — ".join([x for x in [row.get("speaker", ""), row.get("calling", "")] if x]).strip(" —")
    meta = " ".join([str(row.get("year", "")).strip(), str(row.get("season", "")).strip()]).strip()
    title = row.get("title", "")
    url = row.get("url", "")
    snippet = _safe_first_text(row, granularity)
    if snippet and len(snippet) > 220:
        snippet = snippet[:220] + "…"
    lines = [f"- {title} ({meta})", f"  {who}" if who else "  ", f"  {url}"]
    if granularity != "talks":
        lines.append(f"  {snippet}")
    return "\n".join(lines)

# ------------------------ Optional Generation ------------------------

def answer_with_context(question: str, rows: list[dict], model: str = "gpt-4.1-mini") -> str:
    """Call Chat Completions using the top-3 rows as the only context (RAG generation)."""
    try:
        from openai import OpenAI
    except Exception as e:
        print("ERROR: openai package not installed. Try: pip install openai", file=sys.stderr)
        raise
    client = OpenAI(api_key=_load_openai_key())

    sources = []
    for r in rows:
        title = r.get("title", "")
        url = r.get("url", "")
        speaker = r.get("speaker", "")
        year = str(r.get("year", "")).strip()
        text = r.get("text", "")
        if isinstance(text, str):
            try:
                tmp = ast.literal_eval(text)  # handle clusters (list of paragraphs)
                if isinstance(tmp, list):
                    text = "\n".join(tmp)
            except Exception:
                pass
        sources.append(f"### {title} — {speaker} ({year})\n{url}\n{text}\n")

    system = (
        "You are a helpful assistant. Answer ONLY using the provided General Conference excerpts. "
        "Cite the talk titles inline like (Title, Year). If the answer isn't in the excerpts, say so."
    )
    user = f"Question:\n{question}\n\nContext:\n" + "\n\n".join(sources)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2
    )
    return resp.choices[0].message.content

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Semantic search over LDS General Conference (talks/paragraphs/clusters).")
    ap.add_argument("--embedder", choices=["free", "openai"], required=True, help="Which embedding space to query.")
    ap.add_argument("--granularity", choices=["talks", "paragraphs", "clusters"], required=True, help="Search unit.")
    ap.add_argument("--query", required=True, help="User question / search text.")
    ap.add_argument("-k", type=int, default=3, help="Top-k results to display (default 3).")
    ap.add_argument("--generate", action="store_true", help="After retrieval, generate an answer grounded in the top-k.")
    ap.add_argument("--model", default="gpt-4.1-mini", help="Chat model for --generate.")
    args = ap.parse_args()

    df = load_table(args.embedder, args.granularity)
    encoder = pick_encoder(args.embedder)
    qvec = encoder(args.query)

    mat = np.stack(df["embedding"].values)
    sims = cosine_sim(qvec, mat)
    top_idx = np.argsort(-sims)[: args.k]

    print(f"\nTop {args.k} for: “{args.query}”  |  {args.embedder} / {args.granularity}\n")
    rows = []
    for rank, i in enumerate(top_idx, 1):
        row = df.iloc[i].to_dict()
        rows.append(row)
        print(f"{rank}. score={sims[i]:.4f}")
        print(format_row(df.iloc[i], args.granularity))
        print()

    if args.generate:
        print("\n--- Generated answer (grounded in the retrieved talks) ---\n")
        try:
            print(answer_with_context(args.query, rows, model=args.model))
        except Exception as e:
            print(f"Generation failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
