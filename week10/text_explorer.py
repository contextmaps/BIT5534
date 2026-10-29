"""text_explorer.py, BIT 5534 week 10, bag of words, embeddings and BERTopic.

Opens a window like the week 8 tools. Open a .csv or .xlsx file, choose the column
that holds the text, set the number of topics k and of LSA components, and press
Run the analysis. The tool builds the document term matrix (DTM) of raw counts and of
tf-idf weights, finds k topics with NMF, computes LSA scores, places every document
on a 2D map with PCA, t-SNE and UMAP, and opens one interactive page,
<file>_text_explorer.html, beside your file. Save file with them... writes your file
with the topic weights, the LSA scores and the term count added as numeric columns,
which the week 8 tools (REG, TREE, NN, SVM) read as inputs.

    py text_explorer.py                      (Windows)
    python3 text_explorer.py                 (Mac)
    py text_explorer.py --cli tickets.csv --text text --topics 6 --lsa 5   (no window)

Embeddings (Thursday). Choose an embeddings file beside your data, one row per
document, an id column shared with the data and the numbers E1, E2, ... The rows are
matched by that id, so the two files may be in different orders. With an embeddings
file the tool also draws the maps and finds the neighbors on the embeddings, and runs
BERTopic's recipe: UMAP down to five dimensions, HDBSCAN clusters, and c-TF-IDF for
each topic's words. Save file with them... can then add BT_topic, the topic of each
document as a category, and BT_0 to BT_m, its cosine with each topic's center.
To make an embeddings file, install sentence-transformers and press Make embeddings,
or run without the window:
    py text_explorer.py --embed tickets.csv --text text --id ticket_id
which writes tickets_embeddings.csv with all-MiniLM-L6-v2 (the first run downloads
the model, about 90 MB). Without sentence-transformers the tool still reads an
embeddings file made elsewhere.

Libraries: pandas, numpy and scikit-learn, which the week 8 tools already use.
UMAP needs one more, pip install umap-learn. Without it the page offers PCA and
t-SNE and says so. With plotly installed (pip install plotly) the page carries its
plotting library and opens offline; without it the page loads the library from the
web.

The weights are scikit-learn's defaults, written out here so the page and the
lecture can state them exactly:
    count(t, d)  the number of times term t appears in document d
    df(t)        the number of documents that contain t at least once
    idf(t)       ln((1 + N) / (1 + df(t))) + 1, N documents
    tf-idf(t, d) count(t, d) * idf(t), then each document row is scaled to length 1
Topics come from NMF on the tf-idf matrix (k topics, each document a non-negative
mix of them). LSA is a truncated SVD of the same matrix, the decomposition of week
4 without centering. Neighbors are by cosine similarity of the document rows.
"""
import argparse, json, math, os, sys, webbrowser, html as _html
import numpy as np
import pandas as pd

SEED = 5534


def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def text_columns(df):
    """Columns holding strings, longest average text first."""
    obj = [c for c in df.columns if df[c].dtype == object]
    return sorted(obj, key=lambda c: -df[c].astype(str).str.len().mean())


def guess_text_column(df):
    for c in df.columns:
        if str(c).lower() in ("text", "description", "ticket_text", "review", "comment"):
            return c
    obj = text_columns(df)
    return obj[0] if obj else None


def id_column(df, text_col):
    """The first column whose values are all different, used to name documents."""
    for c in df.columns:
        if c != text_col and df[c].is_unique and df[c].notna().all():
            return c
    return None


EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"


def can_embed():
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


def embeddings_path(path):
    return os.path.splitext(path)[0] + "_embeddings.csv"


def make_embeddings(df, text_col, id_col, out_path, model=EMB_MODEL, log=print):
    """Embed every text with a sentence transformer and write one row per document."""
    from sentence_transformers import SentenceTransformer
    texts = df[text_col].fillna("").astype(str).tolist()
    log(f"Loading {model}. The first time, this downloads about 90 MB.")
    m = SentenceTransformer(model, device="cpu")
    log(f"Embedding {len(texts)} documents.")
    E = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    out = pd.DataFrame(np.round(E, 6), columns=[f"E{i + 1}" for i in range(E.shape[1])])
    if id_col is not None:
        out.insert(0, id_col, df[id_col].values)
    out.to_csv(out_path, index=False)
    log(f"Wrote {os.path.basename(out_path)}, {len(out)} rows and {E.shape[1]} numbers each.")
    return out_path


def embedding_columns(E, key=None):
    import re
    cols = [c for c in E.columns if re.fullmatch(r"[Ee]\d+", str(c))]
    if not cols:
        cols = [c for c in E.columns if c != key and pd.api.types.is_numeric_dtype(E[c])]
    return cols


def shared_keys(df, E):
    """Columns in both files whose values are all different, the ones rows can be matched on."""
    return [c for c in df.columns if c in E.columns and df[c].is_unique and E[c].is_unique and df[c].notna().all()]


def align_embeddings(df, E, key):
    """The embedding of every row of df, in df's order, each row scaled to length 1."""
    cols = embedding_columns(E, key)
    if len(cols) < 2:
        raise ValueError("The embeddings file has no columns of numbers named E1, E2, ...")
    if key:
        m = E.drop_duplicates(subset=[key]).set_index(E.drop_duplicates(subset=[key])[key].astype(str))
        idx = df[key].astype(str)
        miss = ~idx.isin(m.index)
        if miss.any():
            ex = ", ".join(idx[miss].head(3))
            raise ValueError(f"{int(miss.sum())} rows of the data have no embedding with the same {key}, for example {ex}. "
                             "Make the embeddings again from this file.")
        X = m.loc[idx.values, cols].to_numpy(float)
    else:
        if len(E) != len(df):
            raise ValueError(f"The files share no id column and have {len(df)} and {len(E)} rows, so rows cannot be matched.")
        X = E[cols].to_numpy(float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def bertopic_recipe(docs, X, min_size=15, seed=SEED, log=print, has_umap=True):
    """BERTopic's three steps, written out: reduce, cluster, describe.

    1. UMAP reduces the embeddings to 5 dimensions (cosine, 15 neighbors, min_dist 0).
    2. HDBSCAN finds dense groups; a document in no group is an outlier, topic -1.
    3. c-TF-IDF: the documents of each topic are joined into one, and each term is
       weighted by tf(t, topic) * ln(1 + A / f(t)), A the average number of words per
       topic and f(t) the term's count over all topics.
    Topics are numbered by size, BT0 the largest, as BERTopic does."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        raise ValueError("BERTopic here needs scikit-learn 1.3 or newer: py -m pip install -U scikit-learn")
    if has_umap:
        import umap
        log("BERTopic step 1, UMAP to 5 dimensions")
        Z = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine",
                      random_state=seed, n_jobs=1).fit_transform(X)
        reducer = "UMAP"
    else:
        from sklearn.decomposition import PCA
        log("BERTopic step 1, PCA to 5 dimensions, because UMAP is not installed")
        Z = PCA(n_components=5, random_state=seed).fit_transform(X)
        reducer = "PCA (UMAP not installed)"
    log(f"BERTopic step 2, HDBSCAN, minimum topic size {min_size}")
    raw = HDBSCAN(min_cluster_size=int(min_size), copy=True).fit_predict(Z)
    ks = sorted(set(raw.tolist()) - {-1}, key=lambda k: (-(raw == k).sum(), int(np.where(raw == k)[0][0])))
    if not ks:
        raise ValueError(f"HDBSCAN found no topic of at least {min_size} documents. Lower the minimum topic size.")
    remap = {k: i for i, k in enumerate(ks)}
    lab = np.array([remap.get(v, -1) for v in raw])
    K = len(ks)
    log(f"BERTopic step 3, c-TF-IDF words for {K} topics")
    cv = CountVectorizer(stop_words="english", token_pattern=TOKEN)
    C = cv.fit_transform(docs)
    vocab = cv.get_feature_names_out()
    classes = list(range(K)) + ([-1] if (lab == -1).any() else [])
    M = np.vstack([np.asarray(C[lab == k].sum(axis=0)).ravel() for k in classes]).astype(float)
    tf = M / np.where(M.sum(1, keepdims=True) == 0, 1, M.sum(1, keepdims=True))
    A = M.sum() / len(classes)
    idf = np.log(1 + A / np.maximum(M.sum(0), 1))
    CT = tf * idf
    width = len(str(K - 1))
    name = lambda k: "outlier" if k < 0 else f"BT{k:0{width}d}"
    topics = []
    for r, k in enumerate(classes):
        top = CT[r].argsort()[::-1][:10]
        topics.append({"id": name(k), "words": [vocab[i] for i in top], "w": [round(float(CT[r, i]), 4) for i in top],
                       "n": int((lab == k).sum())})
    cent = normalize(np.vstack([X[lab == k].mean(0) for k in range(K)]))
    scores = X @ cent.T
    dend = None
    if K >= 3:
        from scipy.cluster.hierarchy import linkage, dendrogram
        Lk = linkage(normalize(CT[:K]), "ward", optimal_ordering=True)
        d = dendrogram(Lk, no_plot=True, labels=[f"{name(k)}: {', '.join(topics[k]['words'][:3])}" for k in range(K)])
        dend = {"ic": np.round(d["icoord"], 3).tolist(), "dc": np.round(d["dcoord"], 4).tolist(), "labels": d["ivl"]}
    return dict(lab=lab, labels=[name(k) for k in lab], K=K, topics=topics, scores=scores, dend=dend,
                reducer=reducer, min_size=int(min_size), names=[name(k) for k in range(K)])


def build(df, text_col, k_topics=6, min_df=2, seed=SEED, want_umap=True, log=print, n_lsa=5, want_tsne=True,
          emb=None, min_topic=15):
    """emb: the embedding of every row, already aligned and scaled, or None."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.decomposition import NMF, TruncatedSVD, PCA
    from sklearn.manifold import TSNE

    docs = df[text_col].fillna("").astype(str).tolist()
    N = len(docs)
    cv = CountVectorizer(stop_words="english", min_df=min_df, token_pattern=TOKEN)
    try:
        C = cv.fit_transform(docs)                   # counts, N x V, sparse
    except ValueError:
        raise ValueError(f"No terms are left in '{text_col}' after stop words and min_df {min_df}. "
                         "Pick a column of text, or lower the minimum.")
    if C.shape[1] < 3:
        raise ValueError(f"Only {C.shape[1]} terms are left in '{text_col}'. Pick a column of text, or lower the minimum.")
    vocab = cv.get_feature_names_out()
    V = len(vocab)
    k_topics = max(1, min(k_topics, V - 1)); n_lsa = max(1, min(n_lsa, V - 1))
    tt = TfidfTransformer()                          # smooth idf, l2 rows
    W = tt.fit_transform(C)
    df_t = np.asarray((C > 0).sum(axis=0)).ravel()
    idf = tt.idf_
    tot = np.asarray(C.sum(axis=0)).ravel()
    log(f"{N} documents, {V} terms kept (stop words removed, min_df {min_df}).")
    log(f"DTM is {N} x {V}, {C.nnz} nonzero cells, {100 * C.nnz / (N * V):.2f} percent full.")

    log(f"NMF with {k_topics} topics.")
    nmf = NMF(n_components=k_topics, init="nndsvda", random_state=seed, max_iter=600)
    Th = nmf.fit_transform(W)
    H = nmf.components_
    share = Th / np.where(Th.sum(1, keepdims=True) == 0, 1, Th.sum(1, keepdims=True))
    dom = Th.argmax(1) + 1
    dom = np.where(Th.max(1) == 0, 0, dom)
    topics = []
    for j in range(k_topics):
        top = H[j].argsort()[::-1][:10]
        topics.append({"id": j + 1, "words": [vocab[i] for i in top], "w": [round(float(H[j, i]), 4) for i in top],
                       "n": int((dom == j + 1).sum())})
    svd = TruncatedSVD(n_components=n_lsa, random_state=seed)
    L = svd.fit_transform(W)
    lsa = []
    for j in range(svd.n_components):
        comp = svd.components_[j]
        pos = comp.argsort()[::-1][:6]
        neg = comp.argsort()[:6]
        lsa.append({"id": j + 1, "var": round(float(svd.explained_variance_ratio_[j]), 4),
                    "pos": [[vocab[i], round(float(comp[i]), 3)] for i in pos],
                    "neg": [[vocab[i], round(float(comp[i]), 3)] for i in neg]})

    # maps
    try:
        import umap  # noqa
        has_umap = want_umap
    except Exception:
        has_umap = False
    maps = {}
    spaces = [("count", C.astype(float)), ("tfidf", W)]
    if emb is not None:
        spaces.append(("emb", emb))
    for wname, M in spaces:
        log(f"PCA on {wname}")
        D = M.toarray() if hasattr(M, "toarray") else M
        maps[f"{wname}|pca"] = PCA(n_components=2, random_state=seed).fit_transform(D)
        if want_tsne:
            log(f"t-SNE on {wname}" + (", which takes a few minutes for this many documents" if N > 5000 else ""))
            red = TruncatedSVD(n_components=min(50, M.shape[1] - 1), random_state=seed).fit_transform(M)
            maps[f"{wname}|tsne"] = TSNE(n_components=2, perplexity=min(30, max(5, N // 4)), init="pca",
                                         random_state=seed).fit_transform(red)
        if has_umap:
            import umap
            log(f"UMAP on {wname}")
            maps[f"{wname}|umap"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                                              random_state=seed, n_jobs=1).fit_transform(M)
        log(f"maps built on {wname}")

    # neighbors by cosine
    neigh = {}
    from sklearn.preprocessing import normalize
    for wname, M in spaces:
        U = normalize(M)
        U = U.tocsr() if hasattr(U, "tocsr") else U
        rows = []
        for a in range(0, N, 1000):                  # in blocks, so a large file fits in memory
            S = U[a:a + 1000] @ U.T
            S = S.toarray() if hasattr(S, "toarray") else np.array(S)
            for r in range(S.shape[0]):
                S[r, a + r] = -1
            k = min(6, N - 1)
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k] if N > k else np.argsort(-S, axis=1)[:, :k]
            for r in range(S.shape[0]):
                o = idx[r][np.argsort(-S[r, idx[r]])]
                rows.append([[int(j), round(float(S[r, j]), 3)] for j in o])
        neigh[wname] = rows

    # per document terms (for the matrix view and shared word highlighting)
    Cc = C.tocsr(); Wc = W.tocsr()
    docterms = []
    for i in range(N):
        s, e = Cc.indptr[i], Cc.indptr[i + 1]
        cols = Cc.indices[s:e]
        docterms.append({vocab[c]: [int(Cc.data[s + n]), round(float(Wc[i, c]), 3)] for n, c in enumerate(cols)})

    feats = pd.DataFrame(index=df.index)
    for j in range(k_topics):
        feats[f"TOPIC_{j + 1}"] = np.round(Th[:, j], 4)
    for j in range(svd.n_components):
        feats[f"LSA_{j + 1}"] = np.round(L[:, j], 4)
    feats["TEXT_terms"] = np.asarray((C > 0).sum(axis=1)).ravel()
    bt = None
    if emb is not None:
        bt = bertopic_recipe(docs, emb, min_topic, seed, log, has_umap=_umap_found())
        feats["BT_topic"] = bt["labels"]
        for j, nm in enumerate(bt["names"]):
            feats[f"BT_{nm[2:]}"] = np.round(bt["scores"][:, j], 4)

    words = sorted(({"t": vocab[i], "df": int(df_t[i]), "idf": round(float(idf[i]), 3), "n": int(tot[i])}
                    for i in range(V)), key=lambda r: -r["n"])
    return dict(N=N, V=V, nnz=int(C.nnz), docs=docs, words=words, topics=topics, lsa=lsa, dom=dom.tolist(),
                maps={k: np.round(v, 3).tolist() for k, v in maps.items()}, has_umap=has_umap,
                neigh=neigh, docterms=docterms, feats=feats, nmf_err=round(float(nmf.reconstruction_err_), 3),
                bt=bt, has_emb=emb is not None)


def _umap_found():
    import importlib.util
    return importlib.util.find_spec("umap") is not None


def page(df, text_col, R, title):
    cats = [c for c in df.columns if c != text_col and 1 < df[c].nunique() <= 12]
    meta = {c: df[c].astype(str).tolist() for c in cats}
    meta["topic (NMF)"] = [f"T{d}" for d in R["dom"]]
    if R.get("bt"):
        meta["topic (BERTopic)"] = R["bt"]["labels"]
    ic = id_column(df, text_col)
    ids = df[ic].astype(str).tolist() if ic is not None else [f"row {i + 1}" for i in range(len(df))]
    payload = dict(title=title, N=R["N"], V=R["V"], nnz=R["nnz"], docs=R["docs"], ids=ids, words=R["words"],
                   topics=R["topics"], lsa=R["lsa"], maps=R["maps"], has_umap=R["has_umap"], neigh=R["neigh"],
                   docterms=R["docterms"], meta=meta, colorby=list(meta.keys()), has_emb=R.get("has_emb", False),
                   bt=({k: R["bt"][k] for k in ("topics", "dend", "reducer", "min_size", "K")} if R.get("bt") else None))
    data = json.dumps(payload, separators=(",", ":"))
    lib = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
    try:
        import plotly
        js = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
        with open(js, encoding="utf-8") as f:
            lib = "<script>" + f.read() + "</script>"
    except Exception:
        pass
    return HTML.replace("__LIB__", lib).replace("__DATA__", data).replace("__TITLE__", _html.escape(title))


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
__LIB__
<style>
:root{--bg:#fbfaf7;--ink:#1f2328;--mute:#5b6470;--line:#d9d6cf;--card:#fff;--acc:#7a1f3d;--acc2:#e87722;--hi:#fff1c2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card)}h1{font-size:19px;margin:0}
.sub{color:var(--mute);font-size:13px}nav{display:flex;gap:6px;padding:10px 20px;flex-wrap:wrap}
nav button{border:1px solid var(--line);background:var(--card);padding:7px 14px;border-radius:7px;cursor:pointer;font-size:14px}
nav button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
section{display:none;padding:6px 20px 30px}section.on{display:block}
.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
label{font-size:13px;color:var(--mute);margin-right:4px}select,input{font-size:14px;padding:4px 6px;margin-right:12px}
#map{width:100%;height:600px}.grow{flex:1 1 560px;min-width:0}.side{flex:0 1 380px;min-width:280px}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid var(--line);padding:3px 7px;text-align:right}th{background:#f1eee8}
td.l,th.l{text-align:left}.z{color:#c9c4ba}.doc{font-size:13.5px;padding:6px 0;border-bottom:1px solid var(--line)}
.sim{font-weight:600;color:var(--acc)}mark{background:var(--hi);padding:0 1px}.scroll{overflow:auto;max-height:560px}
.tbar{display:flex;align-items:center;gap:6px;font-size:13px}.tbar span.b{display:inline-block;height:11px;background:var(--acc2);border-radius:2px}
.topic{min-width:230px}.note{font-size:12.5px;color:var(--mute);max-width:900px}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="summary"></div></header>
<nav id="tabs"><button data-t="map" class="on">Map</button><button data-t="matrix">Matrix</button><button data-t="words">Words</button><button data-t="topics">Topics</button><button data-t="bert">BERTopic</button></nav>
<section id="s-map" class="on">
 <div class="row" style="align-items:center;margin-bottom:8px">
  <span><label>Weights</label><select id="w"><option value="count">counts</option><option value="tfidf" selected>tf-idf</option><option value="emb">embeddings</option></select></span>
  <span><label>Map</label><select id="m"><option value="pca">PCA</option><option value="tsne">t-SNE</option><option value="umap">UMAP</option></select></span>
  <span><label>Color by</label><select id="c"></select></span>
  <span><label>Find</label><input id="find" size="12" placeholder="id"> <button id="findbtn">Find</button> <span id="findmsg" class="note"></span></span>
 </div>
 <div class="row"><div class="card grow"><div id="map"></div></div>
 <div class="card side"><b>Click a point.</b> <span class="note">Its five nearest documents by cosine similarity on the chosen weights or on the embeddings. Words it shares with the clicked document are marked. On the map the five are ringed in green. They are found in all the columns, and a map keeps only two dimensions, so a true neighbor can sit far from the point.</span><div id="nb"></div></div></div>
</section>
<section id="s-matrix"><div class="card"><p class="note">Rows are documents, columns are terms. Each cell shows the count, and under it the tf-idf weight. Pick the documents and the terms shown.</p>
 <div style="margin-bottom:8px"><label>Documents</label><input id="mdocs" size="40"> <label>Terms</label><input id="mterms" size="50"> <button id="mgo">Show</button></div>
 <div class="scroll" id="mat"></div></div></section>
<section id="s-words"><div class="card"><p class="note" id="wnote"></p><div style="margin-bottom:8px"><label>Search</label><input id="wq" size="18"> <label>Sort</label><select id="ws"><option value="n">total count</option><option value="df">document frequency</option><option value="idf">idf</option></select></div><div class="scroll" id="wt"></div></div></section>
<section id="s-topics"><div class="row" id="tp"></div><h3 style="margin:18px 0 6px">Topics against a column</h3><div><label>Column</label><select id="xc"></select></div><div class="card" style="margin-top:8px;display:inline-block" id="xt"></div>
<h3 style="margin:18px 0 6px">LSA, the SVD of the tf-idf matrix</h3><p class="note">Each component is a direction in term space. Loadings are signed, so a component is read from both ends. The matrix is not centered, so LSA 1 points toward the average document and its loadings are nearly all positive.</p><div class="row" id="ls"></div></section>
<section id="s-bert"><p class="note" id="btnote"></p><div class="row" id="bt"></div><h3 style="margin:18px 0 6px">BERTopic topics against a column</h3><div><label>Column</label><select id="bxc"></select></div><div class="card" style="margin-top:8px;display:inline-block" id="bxt"></div>
<h3 style="margin:18px 0 6px">Topic tree</h3><p class="note">Topics joined by how alike their c-TF-IDF words are. Two topics that join low on the tree use similar words.</p><div class="card"><div id="dend" style="width:100%;height:460px"></div></div></section>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const PAL=['#7a1f3d','#e87722','#2b6cb0','#2f855a','#6b46c1','#b7791f','#0f766e','#9b2c2c','#4a5568','#c05621','#2c5282','#276749'];
$('#summary').textContent=`${D.N} documents, ${D.V} terms, DTM ${D.N} x ${D.V} with ${D.nnz} nonzero cells (${(100*D.nnz/(D.N*D.V)).toFixed(2)} percent full).`+(D.has_emb?` Embeddings matched for every document; ${D.bt.K} BERTopic topics.`:' No embeddings file.');
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#tabs button').forEach(x=>x.classList.remove('on'));b.classList.add('on');document.querySelectorAll('section').forEach(s=>s.classList.remove('on'));$('#s-'+b.dataset.t).classList.add('on');if(b.dataset.t==='map')Plotly.Plots.resize('map');});
if(!D.has_umap){const o=$('#m option[value=umap]');o.disabled=true;o.textContent='UMAP (pip install umap-learn)';}
if(!D.maps['tfidf|tsne']){const o=$('#m option[value=tsne]');o.disabled=true;o.textContent='t-SNE (not run)';}
if(!D.has_emb){const o=$('#w option[value=emb]');o.disabled=true;o.textContent='embeddings (choose a file)';}
D.colorby.forEach(c=>{$('#c').add(new Option(c,c));$('#xc').add(new Option(c,c));});
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
const wrap=s=>{let o='',l=0;for(const w of s.split(' ')){if(l+w.length>70){o+='<br>';l=0}o+=w+' ';l+=w.length+1}return o};
let sel=null;
function draw(){const w=$('#w').value,m=$('#m').value,c=$('#c').value;const P=D.maps[w+'|'+m];if(!P)return;
 const lv=[...new Set(D.meta[c])].sort();const tr=lv.map((v,k)=>{const ix=[];D.meta[c].forEach((x,i)=>{if(x===v)ix.push(i)});
 return{type:'scattergl',mode:'markers',name:`${v} (${ix.length})`,x:ix.map(i=>P[i][0]),y:ix.map(i=>P[i][1]),customdata:ix,
 text:ix.map(i=>`<b>${D.ids[i]}</b><br>${wrap(esc(D.docs[i]))}`),hovertemplate:'%{text}<extra></extra>',marker:{size:7,color:PAL[k%PAL.length],opacity:.8}}});
 if(sel!==null){const nbs=D.neigh[w][sel].slice(0,5);
  const lx=[],ly=[];nbs.forEach(([j])=>{lx.push(P[sel][0],P[j][0],null);ly.push(P[sel][1],P[j][1],null)});
  tr.push({type:'scatter',mode:'lines',showlegend:false,x:lx,y:ly,hoverinfo:'skip',line:{color:'#2f855a',width:1.2,dash:'dot'}});
  tr.push({type:'scatter',mode:'markers',name:'five nearest',showlegend:true,x:nbs.map(([j])=>P[j][0]),y:nbs.map(([j])=>P[j][1]),
   text:nbs.map(([j,s],k)=>`neighbor ${k+1}, cosine ${s.toFixed(3)}<br><b>${D.ids[j]}</b>`),hovertemplate:'%{text}<extra></extra>',
   marker:{size:15,color:'rgba(0,0,0,0)',line:{width:2.5,color:'#2f855a'}}});
  tr.push({type:'scatter',mode:'markers',name:'selected',showlegend:true,x:[P[sel][0]],y:[P[sel][1]],hoverinfo:'skip',marker:{size:17,color:'rgba(0,0,0,0)',line:{width:3,color:'#000'}}});}
 Plotly.react('map',tr,{margin:{l:30,r:10,t:10,b:30},legend:{orientation:'h',y:-0.08},xaxis:{zeroline:false,showticklabels:false},yaxis:{zeroline:false,showticklabels:false},hoverlabel:{align:'left'},plot_bgcolor:'#fff'},{responsive:true});
 if(!draw.bound){document.getElementById('map').on('plotly_click',e=>{const i=e.points[0].customdata;if(i!==undefined){sel=i;$('#find').value=D.ids[i];$('#findmsg').textContent='';nb();draw();}});draw.bound=true;}}
function mark(i,j){const a=D.docterms[i],b=D.docterms[j];const sh=new Set(Object.keys(a).filter(t=>t in b));
 return esc(D.docs[j]).replace(/[A-Za-z]+/g,w=>sh.has(w.toLowerCase())?`<mark>${w}</mark>`:w)}
function nb(){if(sel===null)return;const w=$('#w').value;const c=$('#c').value;let h=`<div class="doc"><b>${D.ids[sel]}</b> <i>${c}: ${D.meta[c][sel]}</i><br>${esc(D.docs[sel])}</div>`;
 D.neigh[w][sel].slice(0,5).forEach(([j,s])=>{h+=`<div class="doc"><span class="sim">${s.toFixed(3)}</span> <b>${D.ids[j]}</b> <i>${D.meta[c][j]}</i><br>${mark(sel,j)}</div>`});$('#nb').innerHTML=h;}
['#w','#m','#c'].forEach(s=>$(s).onchange=()=>{draw();nb()});
const IDX=Object.fromEntries(D.ids.map((v,i)=>[String(v).trim().toLowerCase(),i]));
function find(){const q=$('#find').value.trim().toLowerCase();const i=IDX[q];
 if(i===undefined){$('#findmsg').textContent=q?'not found':'';return}
 $('#findmsg').textContent='';sel=i;draw();nb()}
$('#findbtn').onclick=find;$('#find').onkeydown=e=>{if(e.key==='Enter')find()};
draw();
// matrix
$('#mdocs').value=D.ids.slice(0,6).join(' ');
function defaultTerms(ix){const s={};ix.forEach(i=>Object.entries(D.docterms[i]).forEach(([t,v])=>s[t]=(s[t]||0)+v[0]));return Object.keys(s).sort((a,b)=>s[b]-s[a]).slice(0,14)}
function mat(){const ix=$('#mdocs').value.split(/[\s,]+/).map(x=>D.ids.indexOf(x)).filter(i=>i>=0);let tm=$('#mterms').value.trim().toLowerCase().split(/[\s,]+/).filter(Boolean);
 if(!tm.length){tm=defaultTerms(ix);$('#mterms').value=tm.join(' ')}
 let h='<table><tr><th class="l">document</th>'+tm.map(t=>`<th>${t}</th>`).join('')+'</tr>';
 ix.forEach(i=>{h+=`<tr><td class="l" title="${esc(D.docs[i])}"><b>${D.ids[i]}</b><br><span class="note">${esc(D.docs[i].slice(0,60))}...</span></td>`+tm.map(t=>{const v=D.docterms[i][t];return v?`<td>${v[0]}<br><span class="note">${v[1].toFixed(3)}</span></td>`:'<td class="z">0</td>'}).join('')+'</tr>'});
 const wi=Object.fromEntries(D.words.map(r=>[r.t,r]));h+='<tr><th class="l">df / idf</th>'+tm.map(t=>wi[t]?`<th>${wi[t].df}<br>${wi[t].idf.toFixed(3)}</th>`:'<th>not kept</th>').join('')+'</tr></table>';$('#mat').innerHTML=h}
$('#mgo').onclick=mat;mat();
// words
$('#wnote').textContent=`${D.V} terms. df is the number of documents containing the term, idf = ln((1 + ${D.N}) / (1 + df)) + 1.`;
function words(){const q=$('#wq').value.trim().toLowerCase(),k=$('#ws').value;let r=D.words.filter(w=>w.t.includes(q));r=[...r].sort((a,b)=>k==='idf'?a.idf-b.idf:b[k]-a[k]);
 $('#wt').innerHTML='<table><tr><th class="l">term</th><th>total count</th><th>df</th><th>idf</th></tr>'+r.slice(0,400).map(w=>`<tr><td class="l">${w.t}</td><td>${w.n}</td><td>${w.df}</td><td>${w.idf.toFixed(3)}</td></tr>`).join('')+'</table>'}
$('#wq').oninput=words;$('#ws').onchange=words;words();
// BERTopic
if(!D.bt){$('#btnote').textContent='No embeddings file was chosen, so BERTopic did not run. Choose an embeddings file in the window and run the analysis again.';}
else{const B=D.bt;$('#btnote').textContent=`${B.K} topics from BERTopic's recipe on the embeddings: ${B.reducer} to 5 dimensions, HDBSCAN with a minimum topic size of ${B.min_size}, then c-TF-IDF for each topic's words. BT0 is the largest topic. Documents HDBSCAN put in no topic are outliers.`;
 $('#bt').innerHTML=B.topics.map(t=>{const mx=t.w[0]||1;return `<div class="card topic"><b>${t.id}</b> <span class="note">${t.n} documents</span>`+t.words.map((w,k)=>`<div class="tbar"><span style="width:90px">${w}</span><span class="b" style="width:${Math.round(120*t.w[k]/mx)}px"></span></div>`).join('')+'</div>'}).join('');
 D.colorby.forEach(c=>$('#bxc').add(new Option(c,c)));
 const bxt=()=>{const c=$('#bxc').value;const lv=[...new Set(D.meta[c])].sort();const T=D.meta['topic (BERTopic)'];const tl=[...new Set(T)].sort();
  let h='<table><tr><th class="l">topic</th>'+lv.map(v=>`<th>${v}</th>`).join('')+'</tr>';tl.forEach(t=>{h+=`<tr><td class="l">${t}</td>`+lv.map(v=>`<td>${T.filter((x,i)=>x===t&&D.meta[c][i]===v).length}</td>`).join('')+'</tr>'});$('#bxt').innerHTML=h+'</table>'};
 $('#bxc').onchange=bxt;$('#bxc').value=D.colorby.includes('category')?'category':D.colorby[0];bxt();
 if(B.dend){const tr=B.dend.ic.map((x,i)=>({type:'scatter',mode:'lines',x:B.dend.dc[i],y:x,line:{color:'#7a1f3d',width:2},hoverinfo:'skip',showlegend:false}));
  const ticks=B.dend.labels.map((l,i)=>5+10*i);
  const lay={margin:{l:260,r:20,t:10,b:40},xaxis:{title:'distance between topics',zeroline:false},yaxis:{tickvals:ticks,ticktext:B.dend.labels,zeroline:false,showgrid:false},plot_bgcolor:'#fff'};
  const show=()=>Plotly.react('dend',tr,lay,{responsive:true});show();
  document.querySelector('#tabs button[data-t=bert]').addEventListener('click',()=>setTimeout(show,30));}}
// topics
$('#tp').innerHTML=D.topics.map(t=>{const mx=t.w[0];return `<div class="card topic"><b>T${t.id}</b> <span class="note">${t.n} documents</span>`+t.words.map((w,k)=>`<div class="tbar"><span style="width:90px">${w}</span><span class="b" style="width:${Math.round(120*t.w[k]/mx)}px"></span></div>`).join('')+'</div>'}).join('');
function xt(){const c=$('#xc').value;const lv=[...new Set(D.meta[c])].sort();const T=D.meta['topic (NMF)'];const tl=[...new Set(T)].sort();
 let h='<table><tr><th class="l">topic</th>'+lv.map(v=>`<th>${v}</th>`).join('')+'</tr>';tl.forEach(t=>{h+=`<tr><td class="l">${t}</td>`+lv.map(v=>`<td>${T.filter((x,i)=>x===t&&D.meta[c][i]===v).length}</td>`).join('')+'</tr>'});$('#xt').innerHTML=h+'</table>'}
$('#xc').onchange=xt;$('#xc').value=D.colorby.includes('category')?'category':(D.colorby.find(c=>c!=='topic (NMF)')||'topic (NMF)');xt();
$('#ls').innerHTML=D.lsa.map(c=>`<div class="card topic"><b>LSA ${c.id}</b><table style="margin-top:6px"><tr><th class="l">+ end</th><th></th><th class="l">- end</th><th></th></tr>`+c.pos.map((p,k)=>`<tr><td class="l">${p[0]}</td><td>${p[1]}</td><td class="l">${c.neg[k][0]}</td><td>${c.neg[k][1]}</td></tr>`).join('')+'</table></div>').join('');
</script></body></html>"""


def write_page(df, col, R, path):
    hp = os.path.splitext(path)[0] + "_text_explorer.html"
    with open(hp, "w", encoding="utf-8") as f:
        f.write(page(df, col, R, f"Text explorer, {os.path.basename(path)}, column {col}"))
    return hp


def summary_lines(R, col):
    L = [f"Column '{col}': {R['N']} documents, {R['V']} terms, DTM {R['N']} x {R['V']} with {R['nnz']} nonzero cells "
         f"({100 * R['nnz'] / (R['N'] * R['V']):.2f} percent full).", "", "Topics (NMF on tf-idf), the top words of each:"]
    for t in R["topics"]:
        L.append(f"  T{t['id']}  {t['n']:>5} documents   " + ", ".join(t["words"][:8]))
    L += ["", "LSA components (truncated SVD of tf-idf), both ends of each:"]
    for c in R["lsa"]:
        L.append(f"  LSA_{c['id']}  + " + ", ".join(p[0] for p in c["pos"][:5]) + "   - " + ", ".join(n[0] for n in c["neg"][:5]))
    bt = R.get("bt")
    if bt:
        L += ["", f"BERTopic on the embeddings ({bt['reducer']}, HDBSCAN minimum topic size {bt['min_size']}), {bt['K']} topics:"]
        for t in bt["topics"]:
            L.append(f"  {t['id']:8} {t['n']:>5} documents   " + ", ".join(t["words"][:8]))
    return L


def cli(argv):
    ap = argparse.ArgumentParser(description="Bag of words explorer without the window.")
    ap.add_argument("file")
    ap.add_argument("--text", default="", help="the text column; guessed when omitted")
    ap.add_argument("--topics", type=int, default=6)
    ap.add_argument("--lsa", type=int, default=5)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--embeddings", default="", help="an embeddings file to read, matched by --id")
    ap.add_argument("--id", default="", help="the column that matches rows to embeddings")
    ap.add_argument("--min-topic", type=int, default=15, help="BERTopic's minimum topic size")
    a = ap.parse_args(argv)
    df = read_table(a.file)
    col = a.text or guess_text_column(df)
    if col is None:
        sys.exit("No text column found. Name it with --text.")
    emb = None
    if a.embeddings:
        E = read_table(a.embeddings)
        keys = shared_keys(df, E)
        key = a.id or (keys[0] if keys else None)
        emb = align_embeddings(df, E, key)
        print(f"Embeddings from {os.path.basename(a.embeddings)}, {emb.shape[1]} numbers per row, matched on {key or 'row order'}.")
    R = build(df, col, a.topics, a.min_df, want_umap=not a.no_umap, n_lsa=a.lsa, want_tsne=not a.no_tsne,
              emb=emb, min_topic=a.min_topic)
    out = pd.concat([df, R["feats"]], axis=1)
    fp = os.path.splitext(a.file)[0] + "_features.csv"
    out.to_csv(fp, index=False)
    print("\n".join(summary_lines(R, col)))
    hp = write_page(df, col, R, a.file)
    print(f"Wrote {os.path.basename(fp)} and {os.path.basename(hp)}.")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(hp))


def gui(path=None):
    import threading, queue
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import importlib.util
    has_umap = importlib.util.find_spec("umap") is not None   # found, not loaded: loading it is slow

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Text explorer: bag of words, embeddings, topics and maps")
            self.geometry("1200x760")
            self.raw = None; self.path = None; self.R = None; self.col = None; self.hp = None
            self.E = None; self.epath = None
            self.q = queue.Queue(); self.busy = False
            self._build()
            if path:
                self.load(path)
            else:
                self.status.config(text="Open a .csv or .xlsx file to begin.")
                self.after(200, self.open_file)

        def _build(self):
            self.status = ttk.Label(self, text="", anchor="w", padding=(10, 4)); self.status.pack(side="bottom", fill="x")
            # The side panel scrolls, so the window fits a laptop screen.
            outer = ttk.Frame(self); outer.pack(side="left", fill="y")
            cv = tk.Canvas(outer, width=360, highlightthickness=0, borderwidth=0)
            sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
            side = ttk.Frame(cv, padding=10)
            cv.create_window((0, 0), window=side, anchor="nw")
            side.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all"), width=e.width))
            cv.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y"); cv.pack(side="left", fill="y")
            def wheel(e):
                if cv.winfo_height() < side.winfo_reqheight():
                    cv.yview_scroll(-1 if (e.num == 4 or e.delta > 0) else 1, "units")
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                cv.bind_all(ev, lambda e: wheel(e) if str(e.widget).startswith(str(outer)) else None, add="+")
            ttk.Button(side, text="Open file...", command=self.open_file).pack(fill="x")
            self.file_lab = ttk.Label(side, text="No file", wraplength=270, foreground="#555")
            self.file_lab.pack(fill="x", pady=(4, 10))
            ttk.Label(side, text="Text column").pack(anchor="w")
            self.col_var = tk.StringVar()
            self.col_box = ttk.Combobox(side, textvariable=self.col_var, state="readonly", width=32)
            self.col_box.pack(fill="x")
            self.col_box.bind("<<ComboboxSelected>>", lambda e: self.show_column())
            self.col_info = ttk.Label(side, text="", wraplength=270, foreground="#555")
            self.col_info.pack(fill="x", pady=(2, 10))
            g = ttk.Frame(side); g.pack(fill="x")
            self.k_var = tk.IntVar(value=6); self.l_var = tk.IntVar(value=5); self.m_var = tk.IntVar(value=2)
            for r, (lab, var, lo, hi) in enumerate([("Topics, k", self.k_var, 2, 30), ("LSA components", self.l_var, 1, 30),
                                                    ("Keep terms in at least this\nmany documents", self.m_var, 1, 50)]):
                ttk.Label(g, text=lab).grid(row=r, column=0, sticky="w", pady=3)
                ttk.Spinbox(g, from_=lo, to=hi, textvariable=var, width=6).grid(row=r, column=1, sticky="e", padx=(8, 0))
            mf = ttk.LabelFrame(side, text="Maps", padding=6); mf.pack(fill="x", pady=(10, 0))
            ttk.Label(mf, text="PCA is always drawn.").pack(anchor="w")
            self.tsne_var = tk.BooleanVar(value=True); self.umap_var = tk.BooleanVar(value=has_umap)
            ttk.Checkbutton(mf, text="t-SNE", variable=self.tsne_var).pack(anchor="w")
            cb = ttk.Checkbutton(mf, text="UMAP" if has_umap else "UMAP (pip install umap-learn)", variable=self.umap_var)
            cb.pack(anchor="w")
            if not has_umap:
                cb.state(["disabled"])
            ef = ttk.LabelFrame(side, text="Embeddings, for BERTopic (optional)", padding=6); ef.pack(fill="x", pady=(10, 0))
            eb = ttk.Frame(ef); eb.pack(fill="x")
            ttk.Button(eb, text="Choose file...", command=self.choose_emb).pack(side="left", fill="x", expand=True)
            ttk.Button(eb, text="None", width=6, command=self.clear_emb).pack(side="left", padx=(4, 0))
            self.emb_lab = ttk.Label(ef, text="No embeddings file", wraplength=270, foreground="#555")
            self.emb_lab.pack(fill="x", pady=(3, 3))
            kf = ttk.Frame(ef); kf.pack(fill="x")
            ttk.Label(kf, text="Match rows by").pack(side="left")
            self.key_var = tk.StringVar()
            self.key_box = ttk.Combobox(kf, textvariable=self.key_var, state="readonly", width=16)
            self.key_box.pack(side="left", padx=(6, 0))
            mf2 = ttk.Frame(ef); mf2.pack(fill="x", pady=(4, 0))
            ttk.Label(mf2, text="Minimum topic size").pack(side="left")
            self.min_var = tk.IntVar(value=15)
            ttk.Spinbox(mf2, from_=2, to=500, textvariable=self.min_var, width=6).pack(side="right")
            self.make_btn = ttk.Button(ef, text="Make embeddings..." if can_embed() else
                                       "Make embeddings (pip install sentence-transformers)", command=self.make_emb)
            self.make_btn.pack(fill="x", pady=(6, 0))
            if not can_embed():
                self.make_btn.state(["disabled"])
            self.run_btn = ttk.Button(side, text="Run the analysis", command=self.run)
            self.run_btn.pack(fill="x", pady=(12, 0))
            self.page_btn = ttk.Button(side, text="Open the page again", command=self.open_page, state="disabled")
            self.page_btn.pack(fill="x", pady=(4, 0))
            sf = ttk.LabelFrame(side, text="Features for the week 8 tools", padding=6); sf.pack(fill="x", pady=(14, 0))
            self.s_topic = tk.BooleanVar(value=True); self.s_lsa = tk.BooleanVar(value=True); self.s_terms = tk.BooleanVar(value=True)
            self.s_bt = tk.BooleanVar(value=True); self.s_bts = tk.BooleanVar(value=True)
            ttk.Checkbutton(sf, text="Topic weights, TOPIC_1 to TOPIC_k", variable=self.s_topic).pack(anchor="w")
            ttk.Checkbutton(sf, text="LSA scores, LSA_1 to LSA_n", variable=self.s_lsa).pack(anchor="w")
            ttk.Checkbutton(sf, text="Number of terms, TEXT_terms", variable=self.s_terms).pack(anchor="w")
            ttk.Checkbutton(sf, text="BERTopic topic, BT_topic", variable=self.s_bt).pack(anchor="w")
            ttk.Checkbutton(sf, text="BERTopic scores, BT_0 to BT_m", variable=self.s_bts).pack(anchor="w")
            self.save_btn = ttk.Button(sf, text="Save file with them...", command=self.save, state="disabled")
            self.save_btn.pack(fill="x", pady=(6, 0))
            main = ttk.Frame(self, padding=(0, 10, 10, 10)); main.pack(side="left", fill="both", expand=True)
            self.text = tk.Text(main, wrap="word", font=("Consolas", 10), relief="flat", background="#fbfaf7")
            ys = ttk.Scrollbar(main, command=self.text.yview); self.text.configure(yscrollcommand=ys.set)
            ys.pack(side="right", fill="y"); self.text.pack(side="left", fill="both", expand=True)
            self.say("Open a file, choose the column that holds the text, set k, and press Run the analysis.\n"
                     "For BERTopic, choose an embeddings file too. A file named like yours with _embeddings\n"
                     "at the end is chosen for you when it sits beside your file.\n"
                     "The results open as a page in your browser, and a summary appears here.")

        def say(self, s, clear=True):
            if clear:
                self.text.delete("1.0", "end")
            self.text.insert("end", s + "\n"); self.text.see("end")

        def open_file(self):
            p = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.xlsx *.xls"), ("All", "*.*")])
            if p:
                self.load(p)

        def load(self, p):
            try:
                self.raw = read_table(p)
            except Exception as e:
                messagebox.showerror("Cannot read the file", str(e)); return
            self.path = p; self.R = None
            self.file_lab.config(text=f"{os.path.basename(p)}, {len(self.raw)} rows, {self.raw.shape[1]} columns")
            cols = text_columns(self.raw)
            self.col_box["values"] = cols
            g = guess_text_column(self.raw)
            if g is not None:
                self.col_var.set(g)
            self.show_column()
            self.save_btn.state(["disabled"]); self.page_btn.state(["disabled"])
            self.status.config(text=f"Loaded {os.path.basename(p)}.")
            self.clear_emb()
            if os.path.exists(embeddings_path(p)):
                self.attach_emb(embeddings_path(p))

        def show_column(self):
            c = self.col_var.get()
            if self.raw is None or not c:
                self.col_info.config(text=""); return
            s = self.raw[c].fillna("").astype(str)
            w = s.str.split().str.len()
            self.col_info.config(text=f"{s.nunique()} distinct texts, median {int(w.median())} words, longest {int(w.max())}")
            ex = s[s.str.len() > 0].head(3).tolist()
            self.say(f"Column '{c}', first texts:\n\n" + "\n\n".join(e[:300] for e in ex))

        def choose_emb(self):
            if self.raw is None:
                messagebox.showinfo("Open your data first", "Open the file with the text first, then its embeddings."); return
            p = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.xlsx *.xls"), ("All", "*.*")],
                                           initialdir=os.path.dirname(self.path or "."))
            if p:
                self.attach_emb(p)

        def attach_emb(self, p):
            try:
                E = read_table(p)
            except Exception as e:
                messagebox.showerror("Cannot read the embeddings", str(e)); return
            keys = shared_keys(self.raw, E)
            cols = embedding_columns(E, keys[0] if keys else None)
            if len(cols) < 2:
                messagebox.showerror("Not an embeddings file", f"{os.path.basename(p)} has no columns of numbers named E1, E2, ..."); return
            if not keys and len(E) != len(self.raw):
                messagebox.showerror("Rows cannot be matched",
                                     f"{os.path.basename(p)} shares no id column with your data and has {len(E)} rows "
                                     f"against {len(self.raw)}. Make the embeddings again from this file."); return
            self.E, self.epath = E, p
            self.key_box["values"] = keys if keys else ["(row order)"]
            self.key_var.set(keys[0] if keys else "(row order)")
            self.emb_lab.config(text=f"{os.path.basename(p)}, {len(E)} rows, {len(cols)} numbers each")
            self.status.config(text=f"Embeddings chosen: {os.path.basename(p)}, matched on {self.key_var.get()}.")

        def clear_emb(self):
            self.E = None; self.epath = None
            self.emb_lab.config(text="No embeddings file"); self.key_box["values"] = []; self.key_var.set("")

        def make_emb(self):
            if self.busy or self.raw is None or not self.col_var.get():
                return
            col = self.col_var.get(); key = id_column(self.raw, col); out = embeddings_path(self.path)
            if os.path.exists(out) and not messagebox.askyesno("Replace?", f"{os.path.basename(out)} exists. Make it again?"):
                return
            self.busy = True; self.run_btn.state(["disabled"]); self.make_btn.state(["disabled"])
            self.say(f"Making embeddings for '{col}', {len(self.raw)} documents, matched on {key or 'row order'}.")
            self.t0 = __import__("time").time()

            def work():
                try:
                    make_embeddings(self.raw, col, key, out, log=lambda s: self.q.put(("log", s)))
                    self.q.put(("emb", out))
                except Exception as e:
                    self.q.put(("err", str(e)))
            threading.Thread(target=work, daemon=True).start()
            self.after(200, self.poll)

        def run(self):
            if self.busy or self.raw is None or not self.col_var.get():
                return
            self.busy = True; self.run_btn.state(["disabled"])
            self.col = self.col_var.get()
            args = dict(k=int(self.k_var.get()), lsa=int(self.l_var.get()), m=int(self.m_var.get()),
                        t=bool(self.tsne_var.get()), u=bool(self.umap_var.get()) and has_umap, mt=int(self.min_var.get()))
            emb = None
            if self.E is not None:
                key = self.key_var.get()
                try:
                    emb = align_embeddings(self.raw, self.E, None if key in ("", "(row order)") else key)
                except Exception as e:
                    self.busy = False; self.run_btn.state(["!disabled"])
                    messagebox.showerror("Embeddings do not match", str(e)); return
            self.say(f"Running on '{self.col}', {len(self.raw)} documents, k = {args['k']}"
                     + (f", with embeddings, minimum topic size {args['mt']}." if emb is not None else ", no embeddings."))
            self.t0 = __import__("time").time()

            def work():
                try:
                    R = build(self.raw, self.col, args["k"], args["m"], want_umap=args["u"], n_lsa=args["lsa"],
                              want_tsne=args["t"], log=lambda s: self.q.put(("log", s)), emb=emb, min_topic=args["mt"])
                    self.q.put(("log", "Writing the page"))
                    hp = write_page(self.raw, self.col, R, self.path)
                    self.q.put(("done", (R, hp)))
                except Exception as e:
                    self.q.put(("err", str(e)))
            threading.Thread(target=work, daemon=True).start()
            self.after(200, self.poll)

        def poll(self):
            import time
            try:
                while True:
                    kind, v = self.q.get_nowait()
                    if kind == "log":
                        self.say("  " + v, clear=False)
                    elif kind == "emb":
                        self.busy = False; self.run_btn.state(["!disabled"])
                        if can_embed():
                            self.make_btn.state(["!disabled"])
                        self.attach_emb(v)
                        self.say(f"Done in {time.time() - self.t0:.0f} seconds. Press Run the analysis.", clear=False)
                        return
                    elif kind == "err":
                        self.busy = False; self.run_btn.state(["!disabled"])
                        if can_embed():
                            self.make_btn.state(["!disabled"])
                        self.status.config(text="Stopped."); messagebox.showerror("The analysis stopped", v); return
                    else:
                        self.R, self.hp = v
                        self.busy = False; self.run_btn.state(["!disabled"])
                        self.save_btn.state(["!disabled"]); self.page_btn.state(["!disabled"])
                        self.say("\n".join(summary_lines(self.R, self.col)))
                        self.say(f"\nThe page is {os.path.basename(self.hp)}, beside your file.", clear=False)
                        self.status.config(text=f"Done in {time.time() - self.t0:.0f} seconds.")
                        self.open_page(); return
            except queue.Empty:
                pass
            self.status.config(text=f"Working, {time.time() - self.t0:.0f} seconds")
            self.after(300, self.poll)

        def open_page(self):
            if self.hp:
                webbrowser.open("file://" + os.path.abspath(self.hp))

        def save(self):
            if self.R is None:
                return
            f = self.R["feats"]
            keep = [c for c in f.columns if (c.startswith("TOPIC_") and self.s_topic.get())
                    or (c.startswith("LSA_") and self.s_lsa.get()) or (c == "TEXT_terms" and self.s_terms.get())
                    or (c == "BT_topic" and self.s_bt.get())
                    or (c.startswith("BT_") and c != "BT_topic" and self.s_bts.get())]
            if not keep:
                messagebox.showinfo("Nothing to save", "Tick at least one kind of feature."); return
            p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                             initialfile=os.path.splitext(os.path.basename(self.path))[0] + "_features.csv")
            if not p:
                return
            base = self.raw.drop(columns=[c for c in keep if c in self.raw.columns])
            pd.concat([base, f[keep]], axis=1).to_csv(p, index=False)
            self.status.config(text=f"Saved {os.path.basename(p)} with {len(keep)} new columns: {', '.join(keep)}")

    App().mainloop()


def embed_cli(argv):
    ap = argparse.ArgumentParser(description="Make an embeddings file for a text column.")
    ap.add_argument("file")
    ap.add_argument("--text", default="")
    ap.add_argument("--id", default="")
    ap.add_argument("--model", default=EMB_MODEL)
    a = ap.parse_args(argv)
    if not can_embed():
        sys.exit("Making embeddings needs sentence-transformers: py -m pip install sentence-transformers "
                 "(python3 -m pip install sentence-transformers on a Mac).")
    df = read_table(a.file)
    col = a.text or guess_text_column(df)
    key = a.id or id_column(df, col)
    make_embeddings(df, col, key, embeddings_path(a.file), a.model)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--cli":
        cli(argv[1:])
    elif argv and argv[0] == "--embed":
        embed_cli(argv[1:])
    elif "--selftest" in argv:
        cli([a for a in argv if a != "--selftest"] + ["--no-open"])
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        start = argv[0] if argv and os.path.exists(argv[0]) else None
        gui(start)


if __name__ == "__main__":
    try:
        print("Starting the text explorer...", flush=True)
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback, platform
        msg = traceback.format_exc()
        info = f"Python {sys.version} on {platform.platform()}\nexecutable {sys.executable}\n\n"
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_explorer_error.txt"), "w",
                      encoding="utf-8") as f:
                f.write(info + msg)
        except Exception:
            pass
        print(info + msg)
        print("The error above is also saved in text_explorer_error.txt beside the script.")
        try:
            input("Press Enter to close.")
        except Exception:
            pass
