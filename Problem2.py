

import os, json, random, time, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

OUT_DIR = "outputs_p3.2"
os.makedirs(OUT_DIR, exist_ok=True)

# loading names from TraniningName.txt
def load_training_names(filepath="TrainingNames.txt"):
    """Load names from TrainingNames.txt, create if not exists"""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(names)} names from {filepath}")
        return names
    else:
        print(f"ERROR: {filepath} not found! Please create TrainingNames.txt with 1000 Indian names.")
        return []

def save_training_names(names, filepath="TrainingNames.txt"):
    """Save names to TrainingNames.txt"""
    with open(filepath, "w") as f:
        f.write("\n".join(names))
    print(f"Saved {len(names)} names to {filepath}")

# Load names from file
INDIAN_NAMES = load_training_names("TrainingNames.txt")

# If file doesn't exist, raise error (user must provide the file)
if not INDIAN_NAMES:
    raise FileNotFoundError("TrainingNames.txt not found! Please create it with 1000 Indian names.")

# Ensure we have exactly 1000 names (take first 1000 if more, or use as is)
INDIAN_NAMES = INDIAN_NAMES[:1000]
print(f"Using {len(INDIAN_NAMES)} names for training")

#Vocabulary
SOS = "<"
EOS = ">"

def build_vocab(names):
    chars = sorted(set("".join(n.lower() for n in names)) | {SOS, EOS})
    c2i   = {c: i for i, c in enumerate(chars)}
    i2c   = {i: c for c, i in c2i.items()}
    return c2i, i2c

def name_to_tensor(name, c2i):
    seq = [SOS] + list(name.lower()) + [EOS]
    return torch.tensor([c2i[c] for c in seq if c in c2i], dtype=torch.long)


# Model size helper

def model_size_mb(model):
    
    total = sum(p.numel() for p in model.parameters())
    return total * 4 / (1024 ** 2)   # 4 bytes per float32


# TASK 1a – Vanilla RNN (from scratch using nn primitives)

class VanillaRNN(nn.Module):
   
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed   = nn.Embedding(vocab_size, hidden_size)
        self.rnn     = nn.RNN(hidden_size, hidden_size, num_layers=num_layers,
                              batch_first=True, nonlinearity="tanh",
                              dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

        # Initialise weights (Xavier)
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, h=None):
        emb    = self.dropout(self.embed(x))        # (B, T, H)
        out, h = self.rnn(emb, h)                   # (B, T, H)
        logits = self.fc(self.dropout(out))          # (B, T, V)
        return logits, h

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Bilstm
class BidirectionalLSTM(nn.Module):
   
    def __init__(self, vocab_size, hidden_size=128, num_layers=1, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed   = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional encoder: output dim = 2*H
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        # Project 2H → H for decoder initialisation
        self.enc2dec = nn.Linear(2 * hidden_size, hidden_size)

        # Forward decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                               batch_first=True)
        self.fc      = nn.Linear(hidden_size, vocab_size)

        for name, p in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, h=None):
        emb           = self.dropout(self.embed(x))         # (B, T, H)
        enc_out, _    = self.encoder(emb)                   # (B, T, 2H)
        # Context = mean pooled encoder output → projected to H
        context       = enc_out.mean(dim=1, keepdim=True)   # (B, 1, 2H)
        context       = torch.tanh(self.enc2dec(context))   # (B, 1, H)
        # Add context to every decoder timestep
        dec_in        = emb + context.expand_as(emb)
        dec_out, h    = self.decoder(dec_in, h)
        logits        = self.fc(self.dropout(dec_out))      # (B, T, V)
        return logits, h

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Attention RNN
class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v   = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: (B, 1, H),  enc_outputs: (B, T, H)
        energy  = torch.tanh(self.W_h(enc_outputs) + self.W_s(dec_hidden))
        score   = self.v(energy).squeeze(-1)        # (B, T)
        weights = F.softmax(score, dim=-1)           # (B, T)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs)  # (B, 1, H)
        return context, weights


class AttentionRNN(nn.Module):
   
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed     = nn.Embedding(vocab_size, hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.encoder   = nn.RNN(hidden_size, hidden_size, num_layers=num_layers,
                                batch_first=True, nonlinearity="tanh",
                                dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size)
        # Decoder input = embedding + context → 2H
        self.decoder   = nn.RNN(2 * hidden_size, hidden_size,
                                num_layers=num_layers,
                                batch_first=True, nonlinearity="tanh",
                                dropout=dropout if num_layers > 1 else 0)
        self.fc        = nn.Linear(hidden_size, vocab_size)

        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, h=None):
        emb            = self.dropout(self.embed(x))        # (B, T, H)
        enc_out, _     = self.encoder(emb)                  # (B, T, H)

        B, T, H        = enc_out.shape
        dec_hidden     = torch.zeros(self.num_layers, B, H, device=x.device)
        outputs        = []

        for t in range(T):
            q          = dec_hidden[-1].unsqueeze(1)        # (B, 1, H)
            ctx, _     = self.attention(q, enc_out)         # (B, 1, H)
            dec_inp    = torch.cat([emb[:, t:t+1, :], ctx], dim=-1)  # (B,1,2H)
            out, dec_hidden = self.decoder(dec_inp, dec_hidden)
            outputs.append(out)

        out    = torch.cat(outputs, dim=1)                  # (B, T, H)
        logits = self.fc(self.dropout(out))                 # (B, T, V)
        return logits, dec_hidden

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# training
def train_model(model, names, c2i, epochs=45, lr=5e-3, device="cpu"):
    
    
    model.to(device)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.6)
    crit  = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        random.shuffle(names)

        for name in names:
            t = name_to_tensor(name, c2i).to(device)
            if len(t) < 2:
                continue
            x = t[:-1].unsqueeze(0)    # input:  SOS + name chars
            y = t[1:].unsqueeze(0)     # target: name chars + EOS

            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits.squeeze(0), y.squeeze(0))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg = epoch_loss / len(names)
        history.append(avg)
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    return history


# generation
def generate_name(model, c2i, i2c, max_len=15, temperature=1.0, device="cpu"):
    
    model.eval()
    with torch.no_grad():
        inp  = torch.tensor([[c2i[SOS]]], dtype=torch.long, device=device)
        h    = None
        name = []
        for _ in range(max_len):
            logits, h = model(inp, h)
            probs     = F.softmax(logits[0, -1] / temperature, dim=-1)
            idx       = torch.multinomial(probs, 1).item()
            ch        = i2c[idx]
            if ch == EOS:
                break
            if ch != SOS:
                name.append(ch)
            inp = torch.tensor([[idx]], dtype=torch.long, device=device)
    result = "".join(name).strip().capitalize()
    return result if 2 <= len(result) <= 14 else None

def generate_batch(model, c2i, i2c, n=200, temperature=1.0, device="cpu"):
    names, attempts = [], 0
    while len(names) < n and attempts < n * 20:
        g = generate_name(model, c2i, i2c, temperature=temperature, device=device)
        if g:
            names.append(g)
        attempts += 1
    return names


# Task 2 evaluation
def evaluate(generated, training):
    train_set   = set(n.lower() for n in training)
    gen_lower   = [n.lower() for n in generated]
    novel       = [n for n in gen_lower if n not in train_set]
    unique      = set(gen_lower)
    return {
        "total_generated": len(generated),
        "novelty_rate"   : round(len(novel) / max(len(gen_lower),1) * 100, 2),
        "diversity"      : round(len(unique) / max(len(gen_lower),1) * 100, 2),
        "unique_count"   : len(unique),
        "novel_count"    : len(novel),
    }


# Plots
def plot_loss(histories):
    plt.figure(figsize=(10, 5))
    styles = {"VanillaRNN": "b-o", "BLSTM": "r-s", "AttentionRNN": "g-^"}
    for m, hist in histories.items():
        xs = list(range(1, len(hist)+1))
        plt.plot(xs, hist, styles.get(m,"k-"), label=m, linewidth=2, markersize=4,
                 markevery=max(1,len(hist)//10))
    plt.title("Training Loss per Epoch – All Models", fontsize=13)
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "p2_training_loss.png"), dpi=150)
    plt.close()
    print("Saved p2_training_loss.png")

def plot_metrics(metrics):
    models  = list(metrics.keys())
    novelty = [metrics[m]["novelty_rate"] for m in models]
    divers  = [metrics[m]["diversity"]    for m in models]
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, novelty, w, label="Novelty Rate (%)", color="steelblue", edgecolor="k")
    b2 = ax.bar(x + w/2, divers,  w, label="Diversity (%)",    color="tomato",    edgecolor="k")
    ax.set_title("Quantitative Evaluation – Novelty & Diversity", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 115)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (n, d) in enumerate(zip(novelty, divers)):
        ax.text(i - w/2, n + 1.5, f"{n:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, d + 1.5, f"{d:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "p2_metrics.png"), dpi=150)
    plt.close()
    print("Saved p2_metrics.png")

def plot_lengths(gen_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (m, names) in zip(axes, gen_dict.items()):
        lengths = [len(n) for n in names]
        ax.hist(lengths, bins=range(2, 17), color="mediumpurple",
                edgecolor="k", alpha=0.85)
        mean_l = np.mean(lengths)
        ax.axvline(mean_l, color="red", linestyle="--", label=f"Mean={mean_l:.1f}")
        ax.set_title(f"{m}\nName Length Distribution")
        ax.set_xlabel("Length (chars)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "p2_name_lengths.png"), dpi=150)
    plt.close()
    print("Saved p2_name_lengths.png")


# main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print("  CSL7640 – ASSIGNMENT 2 – PROBLEM 2")
    print(f"  Device: {device}")
    print("="*55)

    names = INDIAN_NAMES

    c2i, i2c = build_vocab(names)
    V = len(c2i)
    print(f"\n  Vocab size : {V} chars")
    print(f"  Names      : {len(names)}")

    
    HIDDEN = 128     
    EPOCHS = 80
    LR     = 5e-3

    model_configs = {
        "VanillaRNN"  : VanillaRNN(V,   hidden_size=HIDDEN, num_layers=2, dropout=0.3),
        "BLSTM"       : BidirectionalLSTM(V, hidden_size=HIDDEN, num_layers=1, dropout=0.4),
        "AttentionRNN": AttentionRNN(V,  hidden_size=HIDDEN, num_layers=2, dropout=0.3),
    }

    #Architecture summary 
    print(f"\n  {'Model':<18} {'Hidden':>7} {'Params':>10} {'Size(MB)':>10}")
    print("  " + "-"*48)
    for m, mdl in model_configs.items():
        print(f"  {m:<18} {HIDDEN:>7} {mdl.param_count():>10,} {model_size_mb(mdl):>9.2f}MB")

    # Train, generate, evaluate 
    histories, gen_dict, metrics, samples = {}, {}, {}, {}
    TEMP = {"VanillaRNN": 0.9, "BLSTM": 1.1, "AttentionRNN": 0.9}

    for mname, model in model_configs.items():
        print(f"\n{'─'*55}\n  Training: {mname}\n{'─'*55}")
        t0   = time.time()
        hist = train_model(model, names, c2i, epochs=EPOCHS, lr=LR, device=device)
        print(f"  Done in {time.time()-t0:.0f}s")
        histories[mname] = hist

        print(f"  Generating 200 names (temp={TEMP[mname]}) ...")
        gen              = generate_batch(model, c2i, i2c, n=200,
                                          temperature=TEMP[mname], device=device)
        gen_dict[mname]  = gen
        metrics[mname]   = evaluate(gen, names)
        samples[mname]   = gen[:20]

        m = metrics[mname]
        print(f"  Novelty={m['novelty_rate']}%  Diversity={m['diversity']}%")
        print(f"  Sample : {', '.join(gen[:6])}")

    # Save JSON results 
    with open(os.path.join(OUT_DIR, "p2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUT_DIR, "p2_generated_names.json"), "w") as f:
        json.dump(gen_dict, f, indent=2)

    # Summary table
    print(f"\n{'='*55}")
    print("  QUANTITATIVE EVALUATION SUMMARY")
    print("="*55)
    print(f"  {'Model':<18} {'Novelty%':>9} {'Diversity%':>11} {'Unique':>8}")
    print("  " + "-"*50)
    for mname, m in metrics.items():
        print(f"  {mname:<18} {m['novelty_rate']:>9.1f} "
              f"{m['diversity']:>11.1f} {m['unique_count']:>8}")

    # Submission form answers 
    vrnn = model_configs["VanillaRNN"]
    print(f"\n{'='*55}")
    print("  SUBMISSION FORM ANSWERS")
    print("="*55)
    print(f"\n  P2 – Vanilla RNN params : {vrnn.param_count():,}")
    print(f"  P2 – Vanilla RNN size   : {model_size_mb(vrnn):.2f} MB")
    print(f"\n  P2 – Which model works best?")
    print("  → See comparison below (fill in after seeing your results)")

    #  Plots 
    plot_loss(histories)
    plot_metrics(metrics)
    plot_lengths(gen_dict)

    #  Qualitative samples 
    print(f"\n{'='*55}")
    print("  QUALITATIVE SAMPLES (20 per model)")
    print("="*55)
    for mname, slist in samples.items():
        print(f"\n  {mname}:")
        print("  " + ", ".join(slist))

    print(f"\n{'='*55}")
    print("  ALL DONE – outputs_p2/ folder")
    print("="*55)