

import os, re, json, collections, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.utils import simple_preprocess   # only for tokenization

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


DATA_DIR = "data"
OUT_DIR  = "outputs_final_2.0"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42) # fix randomness for reproducibility

# Hyperparameter experiments
CONFIGS = [

    #  PART 1: Dimension + Window Experiments 
    # (Keep k fixed = 5)

    ("CBOW_d50_w2",   False, 50,  2, 5),
    ("CBOW_d100_w4",  False, 100, 4, 5),

    ("SG_d50_w2",     True,  50,  2, 5),
    ("SG_d100_w4",    True,  100, 4, 5),


    #  PART 2: Negative Sampling Experiments 
    # (Fix best config: dim=100, window=4)

    ("CBOW_d100_w4_k5",   False, 100, 4, 5),
    ("CBOW_d100_w4_k10",  False, 100, 4, 10),
    ("CBOW_d100_w4_k15",  False, 100, 4, 15),

    ("SG_d100_w4_k5",     True, 100, 4, 5),
    ("SG_d100_w4_k10",    True, 100, 4, 10),
    ("SG_d100_w4_k15",    True, 100, 4, 15),
]



EPOCHS    = 10       # keep manageable — scratch impl is slower than gensim
LR        = 0.01    # learning rate
MIN_COUNT = 3       # ignore words appearing fewer than this many times

# Stopwords
STOPWORDS = {
    "the","a","an","of","in","and","to","is","are","was","were","for","on",
    "at","by","with","from","as","be","it","this","that","its","or","but",
    "not","can","also","into","their","which","all","has","have","been","more",
    "will","one","new","s","we","our","about","than","other","such","may",
    "they","them","these","those","some","any","each","both","through","during",
    "before","after","above","between","out","off","over","under","again",
    "then","once","here","there","when","where","while","how","up","i","you",
    "he","she","who","if","do","did","does","had","would","could","should",
    "get","got","no","so","just","even","very","much","many","most",
    "per","his","her","us","via","yet","due","must","within","without",
    "using","used","use","shall","however","therefore","thus","hence","based",
    "given","made","make","along","across","every","either","neither","whether",
    "well","like","since","same","different","first","second","third","two",
    "three","four","five","de","en","la","le","el","please","click","note",
    "web","links","portal","feedback","intranet","contact","home","menu",
    "page","pages","site","login","logout","search","back","next","prev",
    "previous","skip","nav","header","footer","top","bottom","left","right",
    "close","open","show","hide","view","read","loading","redirect","url",
    "http","https","www","html","php","asp","index","default","error",
    "submit","cancel","reset","clear","send","download","upload","warning",
    "iitj","iit","jodhpur","abhiyan","bharat","quicklinks","sitemap",
    "copyright","rights","reserved","powered","developed","maintained",
    "dr","mr","ms","prof","sri","shri","website","digital","india","rka",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
    "nd","rd","th","st","eg","ie","etc","vs","re","co","lt","gt","amp",
    "noc","ay","unnat","uba","dia","rene","vikky","koenigs","sota","redirecttologinpage"}


FALLBACK = """
IIT Jodhpur is a premier technical institute in Rajasthan offering undergraduate
postgraduate and doctoral programs in engineering science and technology.
The department of computer science offers courses in machine learning algorithms
data structures operating systems and natural language processing.
Research scholars and faculty collaborate on sponsored projects publishing in
top conferences and journals in their fields.
Academic regulations govern the examination grading and evaluation process for
all students enrolled in btech mtech and phd programs.
Admission to undergraduate programs is through jee while postgraduate admissions
use gate scores and phd admissions involve written exam and interview.
PhD students must submit a thesis after completing research under faculty supervision.
The institute has laboratories hostels library and computing resources for students.
Engineering students complete final year project as part of degree requirements.
Examinations are held at end of each semester as per the academic calendar.
The scholarship program supports meritorious and economically weaker students.
Internship opportunities are available for students at leading companies and research labs.
Faculty members guide students in research publish papers and mentor scholars.
The curriculum includes courses on algorithms probability statistics and linear algebra.
Graduate students apply for teaching assistantship and research assistantship positions.
Student clubs organize cultural technical and sports activities on campus throughout year.
The placement cell coordinates campus recruitment with companies from various sectors.
Doctoral candidates pass comprehensive exam before beginning dissertation work.
""" * 40

def load_corpus():
    path = os.path.join(DATA_DIR, "corpus.txt")
    if os.path.exists(path):
        text = open(path, encoding="utf-8", errors="ignore").read()
        print(f"Loaded corpus: {len(text):,} chars")
        return text
    print("WARNING: corpus.txt not found, using fallback corpus")
    return FALLBACK

def preprocess(text):
    """Tokenize, lowercase, remove stopwords. Returns list of sentences."""
    text = text.encode("ascii", errors="ignore").decode("ascii")
    sentences_raw = re.split(r"[.\n!?;]", text)
    sentences = []
    for sent in sentences_raw:
        tokens = simple_preprocess(sent, deacc=True, min_len=2, max_len=25)
        tokens = [t for t in tokens if t not in STOPWORDS]
        if len(tokens) >= 3:
            sentences.append(tokens)
    return sentences

def build_vocab(sentences, min_count=MIN_COUNT):
    """Build word->index and index->word mappings."""
    freq = collections.Counter(w for s in sentences for w in s)
    # Keep only words above min_count
    vocab = [w for w, c in freq.items() if c >= min_count]  # remove rare words (frequency < min_count)
    vocab = sorted(vocab)   # deterministic order
    w2i = {w: i for i, w in enumerate(vocab)}# word → index mapping
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w, freq

def corpus_stats(sentences, raw_text, freq, w2i):
    all_tokens = [w for s in sentences for w in s]
    doc_count  = raw_text.count("\n\n") + 1
    stats = {
        "documents":    doc_count,
        "sentences":    len(sentences),
        "total_tokens": len(all_tokens),
        "vocab_size":   len(w2i),
        "top_20_words": freq.most_common(20)
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
        print(f"    {w:<22} {c:>6}")
    print("="*55)
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    return stats

# word cloud
# Optional visualization to inspect most frequent words in corpus.
# Helps validate preprocessing and dataset quality.
def make_wordcloud(freq):
    if not HAS_WORDCLOUD:
        return
    # Filter stopwords from freq for cloud
    clean_freq = {w: c for w, c in freq.items() if w not in STOPWORDS}
    wc = WordCloud(width=900, height=500, background_color="white",
                   colormap="viridis", max_words=120,
                   prefer_horizontal=0.8).generate_from_frequencies(clean_freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud – IIT Jodhpur Corpus", fontsize=16, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "wordcloud.png"), dpi=150)
    plt.close()
    print("Saved wordcloud.png")

# from scratch word to vec

def softmax(x):
    x = x.astype(np.float32)
    e = np.exp(x - np.max(x))
    return e / np.sum(e, dtype=np.float32)

NEG_SAMPLES = 5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_unigram_dist(freq, w2i):
    dist = np.zeros(len(w2i), dtype=np.float32)
    for w, i in w2i.items():
        dist[i] = freq[w]
    dist = dist ** 0.75
    dist /= dist.sum()
    return dist

def generate_cbow_pairs(sentences, w2i, window):
    """
    Generate (context_indices, target_index) pairs for CBOW.
    Context = average of surrounding words, Target = centre word.
    """
    pairs = []
    for sentence in sentences:
        indices = [w2i[w] for w in sentence if w in w2i]
        for i, target in enumerate(indices):
            ctx = []
            for j in range(max(0, i-window), min(len(indices), i+window+1)):
                if j != i:
                    ctx.append(indices[j])
            if ctx:
                pairs.append((ctx, target))
    return pairs

def generate_sg_pairs(sentences, w2i, window):
    """
    Generate (centre_index, context_index) pairs for Skip-gram.
    Centre word predicts each context word.
    """
    pairs = []
    for sentence in sentences:
        indices = [w2i[w] for w in sentence if w in w2i]
        for i, centre in enumerate(indices):
            for j in range(max(0, i-window), min(len(indices), i+window+1)):
                if j != i:
                    pairs.append((centre, indices[j]))
    return pairs


class Word2VecScratch:

    def __init__(self, vocab_size, dim, sg=False, window=2, lr=LR,
                 neg_samples=NEG_SAMPLES, unigram_dist=None):

        self.V = vocab_size  # vocab size 
        self.D = dim # embedding dimension
        self.sg = sg # skip-gram or CBOW
        self.window = window
        self.lr = lr
        self.neg_samples = neg_samples
        self.unigram_dist = unigram_dist

        scale = np.sqrt(2.0 / (vocab_size + dim))
        self.W1 = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)
        self.W2 = np.random.uniform(-scale, scale, (dim, vocab_size)).astype(np.float32)

    def _forward(self, input_indices):
        h = np.mean(self.W1[input_indices], axis=0).astype(np.float32)
        return h

    def _backward_neg(self, input_indices, target_idx, h):

        loss = 0.0

        # POSITIVE
        v_target = self.W2[:, target_idx]
        score = np.dot(h, v_target) # similarity score
        sig = sigmoid(score) # probability

        grad = (sig - 1) # gradient for positive sample
        self.W2[:, target_idx] -= self.lr * grad * h

        dh = grad * v_target  # gradient wrt hidden layer
        loss += -np.log(sig + 1e-10) #  loss contribution

        # NEGATIVE
        neg_samples = np.random.choice(self.V, self.neg_samples, p=self.unigram_dist)

        for neg in neg_samples:
            if neg == target_idx:
                continue

            v_neg = self.W2[:, neg]
            score = np.dot(h, v_neg)
            sig = sigmoid(score)

            grad = sig # gradient for negative sample
            self.W2[:, neg] -= self.lr * grad * h

            dh += grad * v_neg
            loss += -np.log(1 - sig + 1e-10)

        grad_input = dh / len(input_indices)
        for idx in input_indices:
            self.W1[idx] -= self.lr * grad_input

        return loss

    def train(self, sentences, w2i, epochs=EPOCHS):

        print(f"  Generating training pairs ({'SG' if self.sg else 'CBOW'}) ...")

        if self.sg:
            pairs = generate_sg_pairs(sentences, w2i, self.window)
        else:
            pairs = generate_cbow_pairs(sentences, w2i, self.window)

        history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for p in pairs:
                if self.sg:
                    centre, ctx = p
                    h = self._forward([centre])
                    loss = self._backward_neg([centre], ctx, h)
                else:
                    ctx, target = p
                    h = self._forward(ctx)
                    loss = self._backward_neg(ctx, target, h)

                total_loss += loss

            avg_loss = total_loss / len(pairs)
            history.append(avg_loss)
            print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f}")

        return history

    

    
    def similarity(self, w1_idx, w2_idx):
        v1 = self.W1[w1_idx]
        v2 = self.W1[w2_idx]
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def most_similar(self, word, w2i, i2w, topn=5):
        if word not in w2i:
            return []
        idx  = w2i[word]
        vec  = self.W1[idx]
        norm = np.linalg.norm(vec) + 1e-10
        # Compute cosine similarity with all words
        sims = self.W1 @ vec / (np.linalg.norm(self.W1, axis=1) + 1e-10) / norm
        sims[idx] = -1   # exclude self
        top_idxs = np.argsort(sims)[::-1][:topn]
        return [(i2w[i], float(sims[i])) for i in top_idxs]

    def analogy(self, a, b, c, w2i, i2w, topn=3):
        """b - a + c → ? (vector arithmetic)"""
        for w in [a, b, c]:
            if w not in w2i:
                return None, f"OOV: {w}"
        va = self.W1[w2i[a]]
        vb = self.W1[w2i[b]]
        vc = self.W1[w2i[c]]
        target = vb - va + vc
        norm_t = np.linalg.norm(target) + 1e-10
        sims = self.W1 @ target / (np.linalg.norm(self.W1, axis=1) + 1e-10) / norm_t
        # Exclude input words
        for w in [a, b, c]:
            sims[w2i[w]] = -1
        top_idxs = np.argsort(sims)[::-1][:topn]
        results = [(i2w[i], float(sims[i])) for i in top_idxs]
        return results, None

    def get_vector(self, word, w2i):
        return self.W1[w2i[word]] if word in w2i else None



def train_all(sentences, w2i, unigram_dist):
    models   = {}
    all_hist = {}
    results  = []

    print("\n" + "="*55)
    print("  MODEL TRAINING (From Scratch)")
    print("="*55)

    for label, sg, dim, window, k in CONFIGS:
        arch = "SG" if sg else "CBOW"
        print(f"\n  [{label}]  arch={arch}  dim={dim}  window={window}  epochs={EPOCHS}")
        model = Word2VecScratch(
        vocab_size=len(w2i),
        dim=dim,
        sg=sg,
        window=window,
        lr=LR,
        neg_samples=k,   
        unigram_dist=unigram_dist
        )
        history = model.train(sentences, w2i, epochs=EPOCHS)
        models[label]   = model
        np.save(os.path.join(OUT_DIR, f"{label}_W1.npy"), model.W1)
        all_hist[label] = history
        final_loss = history[-1]
        results.append({
        "label": label,
        "arch": arch,
        "dim": dim,
        "window": window,
        "k": k,   
        "lr": LR,
        "epochs": EPOCHS,
        "final_avg_loss": float(round(final_loss, 6))
        })
        print(f"  Final avg loss: {final_loss:.4f}")

    print("\n" + "="*55)
    print(f"  {'Label':<22} {'Arch':<5} {'Dim':<5} {'Win':<4} {'AvgLoss':>10}")
    print("-"*55)
    for r in results:
        print(f"  {r['label']:<22} {r['arch']:<5} {r['dim']:<5} {r['window']:<4} {r['final_avg_loss']:>10.4f}")
    print("="*55)

    with open(os.path.join(OUT_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return models, all_hist, results



QUERY_WORDS = ["research", "student", "phd", "exam",
               "professor", "course", "department"]

ANALOGIES = [
    # (a, b, c, description)   →  b - a + c = ?
    ("undergraduate", "btech",    "postgraduate", "mtech"),
    ("btech",         "student",  "phd",          "scholar"),
    ("professor",     "teaching", "researcher",   "research"),
    ("exam",          "semester", "thesis",       "phd"),
    ("engineering",   "course",   "science",      "lecture"),
]

def semantic_analysis(models, w2i, i2w):
    report = {}

    for mname in [BEST_CBOW, BEST_SG]:
        model = models[mname]
        arch  = "CBOW" if "CBOW" in mname else "Skip-gram"

        print(f"\n{'='*55}")
        print(f"  SEMANTIC ANALYSIS – {arch} ({mname})")
        print("="*55)

        # Nearest neighbours
        nn_results = {}
        print("\n  Top-5 Nearest Neighbours (cosine similarity):")
        for word in QUERY_WORDS:
            nbrs = model.most_similar(word, w2i, i2w, topn=5)
            nn_results[word] = nbrs
            if nbrs:
                nbr_str = ", ".join(f"{w}({s:.3f})" for w, s in nbrs)
                print(f"    {word:<16} → {nbr_str}")
            else:
                print(f"    {word:<16} → not in vocabulary")

        # Analogies
        analogy_results = []
        print("\n  Analogy Experiments (b - a + c = ?):")
        for a, b, c, expected in ANALOGIES:
            res, err = model.analogy(a, b, c, w2i, i2w, topn=3)
            if err:
                print(f"    {a}:{b}::{c}:? — skip ({err})")
                continue
            top_word, top_score = res[0]
            tag = "✓" if top_word == expected else "~"
            print(f"    {b}-{a}+{c} → {top_word} ({top_score:.3f}) {tag}  [expected: {expected}]")
            analogy_results.append({
                "a": a, "b": b, "c": c,
                "expected": expected,
                "got": top_word, "score": top_score,
                "top3": res
            })

        report[mname] = {
            "nearest_neighbours": {w: [(n, float(s)) for n, s in v]
                                   for w, v in nn_results.items()},
            "analogies": analogy_results
        }

    with open(os.path.join(OUT_DIR, "semantic_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved semantic_analysis.json")
    return report



VIZ_WORDS = [
    "btech", "mtech", "phd", "undergraduate", "postgraduate",        # programs
    "course", "exam", "semester", "lecture", "syllabus",              # academics
    "research", "thesis", "laboratory", "project", "publication",     # research
    "student", "faculty", "professor", "scholar",                     # people
    "department", "admission", "campus", "degree", "scholarship",     # institute
]
GROUP_LABELS = {0:"Programs", 1:"Academics", 2:"Research", 3:"People", 4:"Institute"}
GROUP_SIZES  = [5, 5, 5, 4, 5]
COLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]

def _make_groups():
    g = []
    for gid, sz in enumerate(GROUP_SIZES):
        g.extend([gid]*sz)
    return g

def _get_vectors(model, w2i):
    groups_all = _make_groups()
    valid_words, valid_groups, vecs = [], [], []
    for word, gid in zip(VIZ_WORDS, groups_all):
        if word in w2i:
            valid_words.append(word)
            valid_groups.append(gid)
            vecs.append(model.W1[w2i[word]])
    return valid_words, valid_groups, np.array(vecs)

def plot_pca(models, w2i):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mname in zip(axes, [BEST_CBOW, BEST_SG]):
        arch  = "CBOW" if "CBOW" in mname else "Skip-gram"
        words, groups, vecs = _get_vectors(models[mname], w2i)
        if len(vecs) < 5:
            ax.set_title(f"PCA – {arch}\n(insufficient vocab)")
            continue
        pca  = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(vecs)
        for gid in range(5):
            idxs = [i for i, g in enumerate(groups) if g == gid]
            ax.scatter(proj[idxs,0], proj[idxs,1], c=COLORS[gid],
                       label=GROUP_LABELS[gid], s=80,
                       edgecolors="k", linewidths=0.4, alpha=0.85)
        for i, w in enumerate(words):
            ax.annotate(w, (proj[i,0], proj[i,1]), fontsize=7,
                        textcoords="offset points", xytext=(0,5),
                        ha="center", va="bottom")
        ax.set_title(f"PCA 2D – {arch}", fontsize=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.suptitle("PCA: CBOW vs Skip-gram (Scratch)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_visualization.png"), dpi=150)
    plt.close()
    print("Saved pca_visualization.png")

def plot_tsne(models, w2i):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mname in zip(axes, [BEST_CBOW, BEST_SG]):
        arch  = "CBOW" if "CBOW" in mname else "Skip-gram"
        words, groups, vecs = _get_vectors(models[mname], w2i)
        if len(vecs) < 5:
            ax.set_title(f"t-SNE – {arch}\n(insufficient vocab)")
            continue
        perp = min(15, len(vecs)-1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000, init="pca")
        proj = tsne.fit_transform(vecs)
        for gid in range(5):
            idxs = [i for i, g in enumerate(groups) if g == gid]
            ax.scatter(proj[idxs,0], proj[idxs,1], c=COLORS[gid],
                       label=GROUP_LABELS[gid], s=80,
                       edgecolors="k", linewidths=0.4, alpha=0.85)
        for i, w in enumerate(words):
            ax.annotate(w, (proj[i,0], proj[i,1]), fontsize=7,
                        textcoords="offset points", xytext=(0,5),
                        ha="center", va="bottom")
        ax.set_title(f"t-SNE 2D – {arch}", fontsize=12)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.suptitle("t-SNE: CBOW vs Skip-gram (Scratch)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_visualization.png"), dpi=150)
    plt.close()
    print("Saved tsne_visualization.png")

def plot_cosine_heatmap(models, w2i):
    heat_words = ["research","student","phd","exam","faculty",
                  "course","thesis","laboratory","admission","semester"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mname in zip(axes, [BEST_CBOW, BEST_SG]):
        arch  = "CBOW" if "CBOW" in mname else "Skip-gram"
        model = models[mname]
        wlist = [w for w in heat_words if w in w2i]
        n = len(wlist)
        if n < 3:
            ax.set_title(f"Heatmap – {arch}\n(insufficient vocab)")
            continue
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i,j] = model.similarity(w2i[wlist[i]], w2i[wlist[j]])
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(wlist, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(wlist, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Cosine Similarity – {arch}", fontsize=12)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if mat[i,j] > 0.7 else "black")
    fig.suptitle("Cosine Similarity Heatmap: CBOW vs Skip-gram", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cosine_heatmap.png"), dpi=150)
    plt.close()
    print("Saved cosine_heatmap.png")

def plot_loss_curves(all_hist):
    styles = {"CBOW_dim50_w2":"b-","CBOW_dim100_w4":"b--","CBOW_dim100_w2":"b:",
              "SG_dim50_w2":"r-","SG_dim100_w4":"r--","SG_dim100_w2":"r:"}
    plt.figure(figsize=(10, 5))
    for label, hist in all_hist.items():
        plt.plot(range(1, len(hist)+1), hist,
                 styles.get(label,"k-"), label=label, linewidth=2)
    plt.title("Training Loss per Epoch – All Configurations (Scratch)", fontsize=13)
    plt.xlabel("Epoch"); plt.ylabel("Average Negative Sampling Loss ")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=150)
    plt.close()
    print("Saved training_loss.png")

def plot_loss_bar(results):
    labels = [r["label"] for r in results]
    losses = [r["final_avg_loss"] for r in results]
    colors = ["steelblue" if r["arch"]=="CBOW" else "tomato" for r in results]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, losses, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_title("Final Average Training Loss – All Configurations", fontsize=13)
    ax.set_ylabel("Avg Negative Sampling Loss")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    patches = [mpatches.Patch(color="steelblue", label="CBOW"),
               mpatches.Patch(color="tomato",    label="Skip-gram")]
    ax.legend(handles=patches); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_loss_bar.png"), dpi=150)
    plt.close()
    print("Saved training_loss_bar.png")



if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CSL7640 – ASSIGNMENT 2 – PROBLEM 1")
    print("  Word2Vec FROM SCRATCH (NumPy only)")
    print("="*55)

    # Load & preprocess
    raw_text  = load_corpus()
    print("\nPreprocessing ...")
    sentences = preprocess(raw_text)
    w2i, i2w, freq = build_vocab(sentences, min_count=MIN_COUNT)
    with open(os.path.join(OUT_DIR, "vocab.json"), "w") as f:
        json.dump(w2i, f)
    # Stats & word cloud
    corpus_stats(sentences, raw_text, freq, w2i)
    make_wordcloud(freq)

    unigram_dist = create_unigram_dist(freq, w2i)

    # Train all models
    models, all_hist, results = train_all(sentences, w2i, unigram_dist)
    def get_best_models(results):
        best_cbow = min(
            [r for r in results if r["arch"] == "CBOW"],
            key=lambda x: x["final_avg_loss"]
        )["label"]

        best_sg = min(
            [r for r in results if r["arch"] == "SG"],
            key=lambda x: x["final_avg_loss"]
        )["label"]

        return best_cbow, best_sg


    BEST_CBOW, BEST_SG = get_best_models(results)

    print(f"\nBest CBOW: {BEST_CBOW}")
    print(f"Best SG  : {BEST_SG}")

    # Plots
    plot_loss_curves(all_hist)
    plot_loss_bar(results)

    # Semantic analysis
    print("\nRunning semantic analysis ...")
    semantic_analysis(models, w2i, i2w)

    # Visualisations
    print("\nGenerating visualisations ...")
    plot_pca(models, w2i)
    plot_tsne(models, w2i)
    plot_cosine_heatmap(models, w2i)

    print("\n" + "="*55)
    print("  ALL DONE – outputs/ folder")
    print("="*55)