

import os, re, json, collections, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Optional imports (graceful fallback)
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("[WARN] wordcloud not installed – skipping word cloud.")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 0.  Configuration

DATA_DIR   = "data"
OUT_DIR    = "outputs4"
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparameter grid to experiment with
CONFIGS = [
    # (label,         sg, size, window, negative)
    ("CBOW_dim50_w3",   0,  50,   3,    5),
    ("CBOW_dim100_w5",  0, 100,   5,    5),
    ("CBOW_dim100_w3",  0, 100,   3,   10),
    ("SG_dim50_w3",     1,  50,   3,    5),
    ("SG_dim100_w5",    1, 100,   5,    5),
    ("SG_dim100_w3",    1, 100,   3,   10),
]

# Best models (used for analysis)
BEST_CBOW = "CBOW_dim100_w5"
BEST_SG   = "SG_dim100_w5"

ANALYSIS_WORDS = ["research", "student", "phd", "exam", "professor",
                  "course", "department", "faculty", "thesis", "laboratory",
                  "undergraduate", "postgraduate", "admission", "semester",
                  "degree", "engineering", "science", "technology", "campus",
                  "lecture", "project", "internship", "scholarship", "hostel"]


# 1.  Load / fall-back corpus

FALLBACK_CORPUS = """
IIT Jodhpur is a premier technical institute located in Rajasthan India.
The institute offers undergraduate postgraduate and doctoral programs in engineering science and technology.
Students pursuing BTech MTech and PhD programs can engage in cutting edge research.
The department of computer science and engineering offers courses in machine learning natural language processing and artificial intelligence.
Research scholars and faculty members collaborate on sponsored projects and publish in top conferences and journals.
The academic regulations govern the examination grading and evaluation process for all students.
Admission to undergraduate programs is through JEE Advanced while postgraduate admissions use GATE scores.
PhD students are required to submit a thesis after completing their research work under faculty supervision.
The institute has state of the art laboratories hostels and a central library to support student learning.
Faculty profiles highlight their research interests publications and funded projects.
The curriculum includes courses on algorithms data structures operating systems computer networks and database systems.
Engineering students complete a final year project as part of their degree requirements.
Examinations are held at the end of each semester as per the academic calendar.
The scholarship program supports meritorious and economically weaker students.
Internship opportunities are available for students at leading companies and research organizations.
The institute hosts workshops seminars and technical fests throughout the academic year.
Graduate students can apply for teaching assistantship and research assistantship positions.
The department of electrical engineering covers power systems signal processing and communication.
Mechanical engineering students study thermodynamics fluid mechanics and manufacturing processes.
The mathematics department offers courses in linear algebra calculus probability and statistics.
Research areas include robotics computer vision bioinformatics nanotechnology and quantum computing.
The placement cell coordinates campus recruitment drives with companies from various sectors.
Student clubs and societies organize cultural technical and sports activities on campus.
The institute collaborates with international universities for student exchange programs.
Academic performance is evaluated through continuous assessment quizzes assignments and final exams.
The semester system divides the academic year into two main teaching periods with breaks.
Doctoral candidates must pass a comprehensive exam before beginning their dissertation work.
The institute provides high speed internet access and computing resources to all students.
Library resources include journals textbooks and digital databases for research support.
""" * 30   # repeat to get a reasonable vocabulary size

def load_corpus() -> str:
    """Load scraped corpus or fall back to demo corpus."""
    path = os.path.join(DATA_DIR, "corpus.txt")
    if os.path.exists(path):
        with open(path) as f:
            text = f.read()
        print(f"✅ Loaded corpus from {path} ({len(text):,} chars)")
        return text
    else:
        print("⚠️  data/corpus.txt not found – using built-in demo corpus.")
        print("   Run problem1_scrape.py first for real data!")
        return FALLBACK_CORPUS


# 2.  Preprocessing

STOPWORDS = {
    # Standard English stopwords
    "the","a","an","of","in","and","to","is","are","was","were","for","on",
    "at","by","with","from","as","be","it","this","that","its","or","but",
    "not","can","also","into","their","which","all","has","have","been","more",
    "will","one","new","s","we","our","about","than","other","such","may",
    "they","them","these","those","some","any","each","both","through","during",
    "before","after","above","between","out","off","over","under","again",
    "then","once","here","there","when","where","while","how","up","i","you",
    "he","she","we","who","if","do","did","does","had","would","could","should",
    "get","got","no","so","just","even","very","much","many","most",
    "per","his","her","us","via","yet","due","must","within","without",
    "using","used","use","shall","however","therefore","thus","hence","based",
    "given","made","make","along","across","every","either","neither","whether",
    "well","like","since","same","different","first","second","third","two",
    "three","four","five","de","en","la","le","el","please","click","note",
    # Website navigation / UI boilerplate
    "web","links","portal","feedback","intranet","contact","home","menu",
    "page","pages","site","login","logout","search","back","next","prev",
    "previous","skip","navigation","nav","header","footer","sidebar","top",
    "bottom","left","right","close","open","show","hide","view","read",
    "loading","redirecttologinpage","redirect","redirecting","url","http",
    "https","www","html","php","asp","index","default","error","warning",
    "submit","cancel","reset","clear","send","download","upload",
    # IIT Jodhpur specific navigation noise
    "iitj","iit","jodhpur","abhiyan","bharat","quicklinks","quicklink",
    "breadcrumb","sitemap","copyright","rights","reserved",
    "powered","developed","maintained","updated","last","version",
    # Remaining noise from crawl
    "dr","mr","ms","prof","sri","shri","website","digital","india","rka",
    "vikky","uba","koenigs","rene","dia","ay","noc","sota","appropriaterecommendation",
    "redirecttologinpage","bharat","abhiyan","quicklinks","quicklink",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
    "nd","rd","th","st","eg","ie","etc","vs","re","co","lt","gt","amp",
}

def preprocess(text: str):
    """Return list of sentences, each a list of tokens."""
    # Remove non-ascii
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Split into sentences (rough)
    sentences_raw = re.split(r"[.\n!?;]", text)
    sentences = []
    for sent in sentences_raw:
        tokens = simple_preprocess(sent, deacc=True, min_len=2, max_len=25)
        tokens = [t for t in tokens if t not in STOPWORDS]
        if len(tokens) >= 3:
            sentences.append(tokens)
    return sentences

def corpus_stats(sentences: list, raw_text: str) -> dict:
    """Compute and print dataset statistics."""
    all_tokens   = [t for s in sentences for t in s]
    vocab        = set(all_tokens)
    freq         = collections.Counter(all_tokens)
    doc_count    = raw_text.count("\n\n") + 1   # rough doc count
    stats = {
        "documents"    : doc_count,
        "sentences"    : len(sentences),
        "total_tokens" : len(all_tokens),
        "vocab_size"   : len(vocab),
        "top_20_words" : freq.most_common(20)
    }
    print("\n" + "="*55)
    print("  DATASET STATISTICS")
    print("="*55)
    print(f"  Documents  (approx.) : {stats['documents']:>8,}")
    print(f"  Sentences            : {stats['sentences']:>8,}")
    print(f"  Total Tokens         : {stats['total_tokens']:>8,}")
    print(f"  Vocabulary Size      : {stats['vocab_size']:>8,}")
    print("  Top-20 Words:")
    for w, c in stats["top_20_words"]:
        print(f"    {w:<20} {c:>6}")
    print("="*55)
    return stats


# 3.  Word Cloud

def make_wordcloud(sentences: list):
    if not HAS_WORDCLOUD:
        return
    all_tokens = [t for s in sentences for t in s]
    freq = collections.Counter(all_tokens)
    wc = WordCloud(
        width=900, height=500,
        background_color="white",
        colormap="viridis",
        max_words=120,
        prefer_horizontal=0.8,
    ).generate_from_frequencies(dict(freq))
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud – IIT Jodhpur Corpus", fontsize=16, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "wordcloud.png"), dpi=150)
    plt.close()
    print("✅ Saved wordcloud.png")


# 4.  Train all Word2Vec configurations

def train_all(sentences: list) -> dict:
    """
    Train every configuration in CONFIGS.
    Returns dict: label -> Word2Vec model.
    """
    models = {}
    results_table = []

    print("\n" + "="*55)
    print("  MODEL TRAINING")
    print("="*55)
    print(f"  {'Label':<22} {'Arch':<6} {'Dim':<5} {'Win':<5} {'Neg':<5} {'Loss':>8}")
    print("-"*55)

    for label, sg, size, window, negative in CONFIGS:
        arch = "SG" if sg else "CBOW"
        model = Word2Vec(
            sentences,
            vector_size  = size,
            window       = window,
            min_count    = 2,        # ignore very rare tokens
            sg           = sg,       # 0=CBOW, 1=Skip-gram
            negative     = negative,
            epochs       = 20,
            workers      = 4,
            compute_loss = True,
            seed         = 42,
        )
        loss = model.get_latest_training_loss()
        models[label] = model

        row = f"  {label:<22} {arch:<6} {size:<5} {window:<5} {negative:<5} {loss:>8.2f}"
        print(row)
        results_table.append({
            "label": label, "arch": arch,
            "dim": size, "window": window,
            "negative": negative, "loss": loss
        })

    print("="*55)
    # Save table as JSON for report
    with open(os.path.join(OUT_DIR, "training_results.json"), "w") as f:
        json.dump(results_table, f, indent=2)
    return models

# 5.  Semantic Analysis

REQUIRED_WORDS = ["research", "student", "phd", "exam"]

ANALOGIES = [
    # (a, b, c, expected_d_description)
    ("ug",   "btech",   "pg",        "MTech or MSc"),
    ("phd",  "thesis",  "btech",     "project"),
    ("exam", "grade",   "research",  "publication"),
    ("lab",  "experiment", "class",  "lecture"),
    ("student", "learn", "faculty",  "teach"),
]

def semantic_analysis(models: dict):
    """Nearest neighbours and analogy experiments for both best models."""
    report = {}
    for mname in [BEST_CBOW, BEST_SG]:
        model = models[mname]
        vocab_set = set(model.wv.index_to_key)
        arch = "CBOW" if "CBOW" in mname else "Skip-gram"

        print(f"\n{'='*55}")
        print(f"  SEMANTIC ANALYSIS – {arch} ({mname})")
        print("="*55)

        #  Nearest neighbours
        nn_results = {}
        query_words = ["research", "student", "phd", "exam", "professor", "course", "department"]
        print("\n  Top-5 Nearest Neighbours (cosine similarity):")
        for word in query_words:
            if word not in vocab_set:
                print(f"    '{word}' not in vocabulary – skipping")
                nn_results[word] = []
                continue
            nbrs = model.wv.most_similar(word, topn=5)
            nn_results[word] = nbrs
            nbr_str = ", ".join(f"{w}({s:.3f})" for w, s in nbrs)
            print(f"    {word:<14} → {nbr_str}")

        # Analogies
        analogy_results = []
        print("\n  Analogy Experiments (a : b :: c : ?):")
        for a, b, c, expected in ANALOGIES:
            missing = [w for w in [a, b, c] if w not in vocab_set]
            if missing:
                print(f"    {a}:{b}::{c}:? — skip (OOV: {missing})")
                continue
            try:
                result = model.wv.most_similar(
                    positive=[b, c], negative=[a], topn=3
                )
                top = result[0][0]
                score = result[0][1]
                semantic = "✓ meaningful" if top == expected else f"got '{top}'"
                print(f"    {a}:{b} :: {c}:? → {top} ({score:.3f})  [{semantic}]")
                analogy_results.append({
                    "a": a, "b": b, "c": c,
                    "expected": expected, "got": top, "score": score
                })
            except Exception as e:
                print(f"    {a}:{b}::{c}:? — error: {e}")

        report[mname] = {"nearest_neighbours": nn_results, "analogies": analogy_results}

    with open(os.path.join(OUT_DIR, "semantic_analysis.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Saved semantic_analysis.json")


# 6.  Visualisation – PCA and t-SNE

VIZ_WORDS = [
    # Grouped for colour-coding
    # Group 0: programs
    "btech", "mtech", "phd", "ug", "pg", "undergraduate", "postgraduate",
    # Group 1: academics
    "course", "exam", "grade", "semester", "lecture", "syllabus",
    # Group 2: research
    "research", "thesis", "publication", "lab", "experiment", "project",
    # Group 3: people
    "student", "faculty", "professor", "scholar", "researcher",
    # Group 4: institute
    "department", "campus", "hostel", "library", "admission",
]

GROUP_LABELS = {0: "Programs", 1: "Academics", 2: "Research",
                3: "People", 4: "Institute"}
GROUP_SIZES  = [7, 6, 6, 5, 5]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

def _make_groups():
    groups = []
    idx = 0
    for gid, sz in enumerate(GROUP_SIZES):
        groups.extend([gid]*sz)
    return groups

def _get_vectors(model, words):
    vocab = set(model.wv.index_to_key)
    valid_words  = [w for w in words if w in vocab]
    valid_groups = [g for w, g in zip(words, _make_groups()) if w in vocab]
    vecs = np.array([model.wv[w] for w in valid_words])
    return valid_words, valid_groups, vecs


def plot_pca(models: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mname in zip(axes, [BEST_CBOW, BEST_SG]):
        model  = models[mname]
        arch   = "CBOW" if "CBOW" in mname else "Skip-gram"
        words, groups, vecs = _get_vectors(model, VIZ_WORDS)
        if len(vecs) < 5:
            ax.set_title(f"PCA – {arch}\n(insufficient vocab)")
            continue

        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(vecs)

        for gid in range(5):
            idxs = [i for i, g in enumerate(groups) if g == gid]
            ax.scatter(proj[idxs, 0], proj[idxs, 1],
                       c=COLORS[gid], label=GROUP_LABELS[gid],
                       s=80, edgecolors="k", linewidths=0.4, alpha=0.85)

        # Annotate every word
        for i, w in enumerate(words):
            ax.annotate(w, (proj[i, 0], proj[i, 1]),
                        fontsize=7, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5))

        ax.set_title(f"PCA 2D Projection – {arch}", fontsize=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)

    fig.suptitle("PCA Visualisation: CBOW vs Skip-gram Embeddings", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_visualization.png"), dpi=150)
    plt.close()
    print(" Saved pca_visualization.png")


def plot_tsne(models: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mname in zip(axes, [BEST_CBOW, BEST_SG]):
        model  = models[mname]
        arch   = "CBOW" if "CBOW" in mname else "Skip-gram"
        words, groups, vecs = _get_vectors(model, VIZ_WORDS)
        if len(vecs) < 5:
            ax.set_title(f"t-SNE – {arch}\n(insufficient vocab)")
            continue

        perplexity = min(15, len(vecs) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=42, max_iter=1000, init="pca")
        proj = tsne.fit_transform(vecs)

        for gid in range(5):
            idxs = [i for i, g in enumerate(groups) if g == gid]
            ax.scatter(proj[idxs, 0], proj[idxs, 1],
                       c=COLORS[gid], label=GROUP_LABELS[gid],
                       s=80, edgecolors="k", linewidths=0.4, alpha=0.85)

        for i, w in enumerate(words):
            ax.annotate(w, (proj[i, 0], proj[i, 1]),
                        fontsize=7, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5))

        ax.set_title(f"t-SNE 2D Projection – {arch}", fontsize=12)
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)

    fig.suptitle("t-SNE Visualisation: CBOW vs Skip-gram Embeddings", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_visualization.png"), dpi=150)
    plt.close()
    print(" Saved tsne_visualization.png")


def plot_cosine_heatmap(models: dict):
    """Cosine similarity heatmap for a core set of words."""
    heat_words = ["research", "student", "phd", "exam", "faculty",
                  "course", "thesis", "lab", "admission", "semester"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mname in zip(axes, [BEST_CBOW, BEST_SG]):
        model = models[mname]
        arch  = "CBOW" if "CBOW" in mname else "Skip-gram"
        vocab = set(model.wv.index_to_key)
        wlist = [w for w in heat_words if w in vocab]
        n = len(wlist)
        if n < 3:
            ax.set_title(f"Heatmap – {arch}\n(insufficient vocab)")
            continue
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = model.wv.similarity(wlist[i], wlist[j])

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(wlist, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(wlist, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Cosine Similarity – {arch}", fontsize=12)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if mat[i,j] < 0.7 else "white")

    fig.suptitle("Cosine Similarity Heatmap: CBOW vs Skip-gram", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cosine_heatmap.png"), dpi=150)
    plt.close()
    print(" Saved cosine_heatmap.png")


def plot_training_loss(models: dict):
    """Bar chart of final training losses across all configs."""
    with open(os.path.join(OUT_DIR, "training_results.json")) as f:
        results = json.load(f)

    labels  = [r["label"] for r in results]
    losses  = [r["loss"]  for r in results]
    colors  = ["steelblue" if r["arch"]=="CBOW" else "tomato" for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, losses, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_title("Final Training Loss – All Configurations", fontsize=13)
    ax.set_ylabel("Cumulative Training Loss")
    ax.set_xlabel("Model Configuration")
    plt.xticks(rotation=35, ha="right", fontsize=9)

    legend_patches = [
        mpatches.Patch(color="steelblue", label="CBOW"),
        mpatches.Patch(color="tomato",    label="Skip-gram"),
    ]
    ax.legend(handles=legend_patches)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=150)
    plt.close()
    print(" Saved training_loss.png")


# ────────────────────────────────────────────────────────────────────────────────
# 7.  Save models
# ────────────────────────────────────────────────────────────────────────────────
def save_models(models: dict):
    model_dir = os.path.join(OUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    for label, model in models.items():
        path = os.path.join(model_dir, f"{label}.model")
        model.save(path)
    print(f" Saved {len(models)} models to {model_dir}/")


# ────────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CSL7640 – ASSIGNMENT 2 – PROBLEM 1")
    print("="*55)

    # Load corpus
    raw_text  = load_corpus()

    # Preprocess
    print("\n Preprocessing ...")
    sentences = preprocess(raw_text)

    # Stats + Word Cloud
    stats = corpus_stats(sentences, raw_text)
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)

    make_wordcloud(sentences)

    # Train models
    print("\n Training models ...")
    models = train_all(sentences)
    save_models(models)

    # Plot training loss
    plot_training_loss(models)

    # Semantic analysis
    print("\n Running semantic analysis ...")
    semantic_analysis(models)

    # Visualisations
    print("\n Generating visualisations ...")
    plot_pca(models)
    plot_tsne(models)
    plot_cosine_heatmap(models)

    print("\n" + "="*55)
    print("  ALL DONE – Check the 'outputs/' folder")
    print("="*55)