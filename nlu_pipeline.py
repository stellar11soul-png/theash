#!/usr/bin/env python3
"""
Hindi / Hinglish intent classification (56 classes, unweighted macro-F1).

Pipeline
  1. Script handling: Devanagari rows are romanized into Hinglish-style spelling
     (deterministic version + noisy spelling variants used as augmentation).
  2. Grouped, stratified K-fold: rows that are near-duplicates (e.g. a sentence and its
     transliteration) always land in the same fold, so OOF scores are not inflated.
  3. Model A: char+word TF-IDF -> LinearSVC on single-script (romanized) text. CPU only.
  4. Model B: fine-tuned MuRIL (default) or any BERT/XLM-R model, with augmentation, label
     smoothing, layer-wise LR decay, and test-time augmentation (original + romanized).
  5. Blend A+B on out-of-fold predictions, optional pseudo-labeling rounds on test text,
     then per-class logit biases tuned for macro-F1 (applied only if they help on held-out groups).
  6. Writes submission.csv (example_id,target).

Parameter budget: ONE google/muril-base-cased is ~237M params and the TF-IDF SVM adds ~6M, so a single
model + SVM is ~244M (<= 300M). NOTE: this script averages 5 folds x 2 seeds = 10 transformers, which is
fine for measuring OOF scores and producing pseudo-labels but would NOT satisfy a 300M limit that covers the
whole submitted pipeline. For a compliant final model use train_final.py (one transformer + SVM).

Install
  pip install torch transformers indic-transliteration scikit-learn pandas scipy

Run (GPU, e.g. Colab T4; ~1 hour for 5 folds x 2 seeds x 2 rounds)
  python nlu_pipeline.py --data_dir /path/to/csvs --out_dir out
CPU-only quick baseline (no torch needed)
  python nlu_pipeline.py --data_dir /path/to/csvs --out_dir out_cpu --skip_transformer
If MuRIL cannot be downloaded:  --model xlm-roberta-base
Resume after a disconnect: rerun the same command (fold results are cached in out_dir/cache).
Use --fresh to ignore the cache after changing hyper-parameters.
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC

DEV_RE = re.compile(r"[\u0900-\u097F]")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------------------
# Text normalisation / romanisation
# --------------------------------------------------------------------------------------
def is_dev(s):
    return bool(DEV_RE.search(s))


def norm(s):
    """Lowercase Latin text, drop punctuation, collapse spaces. (Only used on Latin text.)"""
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=None)
def _itrans(word):
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    return transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)


# (deterministic spelling, probability of the *short* spelling in noisy mode)
_LONG = {"A": ("aa", "a", 0.45), "I": ("i", "ee", 0.20), "U": ("oo", "u", 0.40)}
_RULES = ((".N", "n"), ("~N", "n"), (".h", ""), (".n", "n"), ("~n", "n"), ("M", "n"), ("N", "n"),
          ("H", "h"), ("Ch", "chh"), ("Sh", "sh"), ("T", "t"), ("D", "d"))
_C = "bcdfghjklmnpqrstvwxyz"
_MEDIAL_SCHWA = re.compile(rf"(?<=[aeiouAEIOU])([{_C}]h?)a(?=[{_C}]h?[aeiouAEIOU])")


def _rom_word(w, rng):
    r = re.sub(r"RR?i", "ri", _itrans(w.replace("\u0911", "\u0913").replace("\u0949", "\u094b")))
    # anusvara after e: mein / karein (but den)
    if r == "meM":
        r = "mein"
    elif r.endswith("eM") and len(r) > 3:
        r = r[:-2] + ("ein" if rng is None or rng.random() < 0.7 else "en")
    for a, b in _RULES:
        r = r.replace(a, b)
    # Hindi schwa deletion: ghara -> ghar (final), karane -> karne (medial); noisy mode skips it sometimes
    if (len(r) >= 2 and r.endswith("a") and r[-2] in _C and re.search("[aeiouAEIOU]", r[:-1])
            and (rng is None or rng.random() < 0.85)):
        r = r[:-1]
    if rng is None or rng.random() < 0.85:
        r = _MEDIAL_SCHWA.sub(r"\1", r)

    def pick(m):
        det, alt, p_short = _LONG[m.group(0)]
        final = m.end() == len(m.string)
        if m.group(0) == "A" and final:      # word-final long a is usually spelled 'a' (kya, tha, mera)
            return "a" if rng is None or rng.random() < 0.85 else "aa"
        if rng is None:
            return det
        short, long_ = (alt, det) if len(alt) < len(det) else (det, alt)
        return short if rng.random() < p_short else long_

    r = re.sub(r"[AIU]", pick, r)
    return r.replace("E", "e").replace("O", "o").lower()


def rom(text, rng=None):
    """Devanagari -> Hinglish-style Roman text. rng=None: deterministic. rng given: noisy variant."""
    out = [(_rom_word(p, rng) if DEV_RE.match(p) else p) for p in re.split(r"([\u0900-\u097F]+)", text)]
    s = "".join(out)
    if rng is not None and rng.random() < 0.3:
        s = s.replace("v", "w")
    return norm(s)


class Data:
    """Holds raw text, script flag, normalised single-script text and noisy variants for a dataframe."""

    def __init__(self, texts, n_aug, seed):
        rng = random.Random(seed)
        self.text = [str(t) for t in texts]
        self.dev = np.array([is_dev(t) for t in self.text])
        self.norm = [rom(t) if d else norm(t) for t, d in zip(self.text, self.dev)]
        self.noisy = [[rom(t, rng) for _ in range(n_aug)] if d else [] for t, d in zip(self.text, self.dev)]


def gather(D_tr, tri, y, D_te, pidx, py, n_aug, use_norm):
    """Training texts/labels for one fold: fold-train rows (+noisy variants) and pseudo-labelled test rows."""
    texts, labels = [], []
    for D, idx, yy in ((D_tr, tri, y[tri]), (D_te, pidx, py)):
        for i, lab in zip(idx, yy):
            texts.append(D.norm[i] if use_norm else D.text[i])
            labels.append(lab)
            for v in D.noisy[i][:n_aug]:
                texts.append(v)
                labels.append(lab)
    return texts, np.array(labels, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Grouped folds
# --------------------------------------------------------------------------------------
def nn_sim(Xq, Xp, chunk=500):
    XpT = Xp.T.tocsr()
    out = np.zeros(Xq.shape[0], dtype=np.float32)
    for s in range(0, Xq.shape[0], chunk):
        out[s:s + chunk] = (Xq[s:s + chunk] @ XpT).toarray().max(axis=1)
    return out


def make_groups(texts, thr, vec):
    X = vec.transform(texts).tocsr()
    XT = X.T.tocsr()
    n = X.shape[0]
    I, J, V = [], [], []
    for s in range(0, n, 500):
        S = (X[s:s + 500] @ XT).toarray()
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = 0
        ii, jj = np.nonzero(S > thr)
        I.append(ii + s)
        J.append(jj)
        V.append(S[ii, jj])
    I, J, V = np.concatenate(I), np.concatenate(J), np.concatenate(V)
    t = thr
    while True:
        keep = V > t
        g = csr_matrix((np.ones(keep.sum()), (I[keep], J[keep])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        biggest = np.bincount(lab).max()
        if biggest <= max(50, 0.02 * n) or t >= 0.95:
            break
        t += 0.05
    log(f"groups: sim threshold {t:.2f} -> {lab.max() + 1} groups, largest {biggest} rows, "
        f"{(np.bincount(lab) > 1).sum()} multi-row groups")
    return lab, X


# --------------------------------------------------------------------------------------
# Metrics / calibration helpers
# --------------------------------------------------------------------------------------
def macro_f1(y, p, K):
    tp = np.bincount(y[y == p], minlength=K).astype(float)
    denom = np.bincount(p, minlength=K) + np.bincount(y, minlength=K)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1[denom > 0].mean()


def per_class_f1(y, p, K):
    tp = np.bincount(y[y == p], minlength=K).astype(float)
    denom = np.bincount(p, minlength=K) + np.bincount(y, minlength=K)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def fit_temperature(scores, y):
    best = (1e9, 1.0)
    for T in (0.5, 1, 2, 3, 4, 6, 8, 12, 16):
        lp = log_softmax(scores * T, axis=1)
        nll = -lp[np.arange(len(y)), y].mean()
        if nll < best[0]:
            best = (nll, T)
    return best[1]


def tune_bias(lp, y, K, grid=np.linspace(-1.5, 1.5, 13), rounds=2):
    b = np.zeros(K)
    best = macro_f1(y, lp.argmax(1), K)
    for _ in range(rounds):
        for c in range(K):
            keep = b[c]
            for g in grid:
                b[c] = g
                s = macro_f1(y, (lp + b).argmax(1), K)
                if s > best + 1e-9:
                    best, keep = s, g
            b[c] = keep
    return b, best


def bias_heldout_gain(lp, y, groups, K):
    rng = np.random.RandomState(0)
    ug, inv = np.unique(groups, return_inverse=True)
    A = (rng.rand(len(ug)) < 0.5)[inv]
    gains = []
    for tune, ev in ((A, ~A), (~A, A)):
        b, _ = tune_bias(lp[tune], y[tune], K)
        base = macro_f1(y[ev], lp[ev].argmax(1), K)
        gains.append(macro_f1(y[ev], (lp[ev] + b).argmax(1), K) - base)
    return float(np.mean(gains))


def report(name, lp, y, dev_mask, K):
    p = lp.argmax(1)
    f = per_class_f1(y, p, K)
    log(f"{name}: macro-F1 {macro_f1(y, p, K):.4f} | Devanagari {macro_f1(y[dev_mask], p[dev_mask], K):.4f} | "
        f"romanized {macro_f1(y[~dev_mask], p[~dev_mask], K):.4f}")
    worst = np.argsort(f)[:8]
    log("   worst classes:", ", ".join(f"{c}:{f[c]:.2f}" for c in worst))
    return macro_f1(y, p, K)


# --------------------------------------------------------------------------------------
# Model A: TF-IDF + LinearSVC
# --------------------------------------------------------------------------------------
def tfidf_stage(D_tr, D_te, y, folds, K, args, pidx, py):
    oof = np.zeros((len(y), K), np.float32)
    te = np.zeros((len(D_te.text), K), np.float32)
    for k, (tri, vai) in enumerate(folds):
        texts, labels = gather(D_tr, tri, y, D_te, pidx, py, args.tfidf_aug, use_norm=True)
        vec = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2),
                         TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True))
        X = vec.fit_transform(texts)
        m = LinearSVC(C=args.svm_c, class_weight="balanced").fit(X, labels)
        oof[vai] = m.decision_function(vec.transform([D_tr.norm[i] for i in vai]))
        te += m.decision_function(vec.transform(D_te.norm)) / len(folds)
        log(f"  tfidf fold {k + 1}/{len(folds)} done")
    return oof, te


# --------------------------------------------------------------------------------------
# Model B: transformer
# --------------------------------------------------------------------------------------
def layerwise_groups(model, lr, decay, wd):
    n_layers = model.config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        lid = int(m.group(1)) + 1 if m else (0 if "embeddings" in name else n_layers + 1)
        no_wd = name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name
        groups.setdefault((lid, no_wd), []).append(p)
    return [{"params": ps, "lr": lr * decay ** (n_layers + 1 - lid), "weight_decay": 0.0 if nw else wd}
            for (lid, nw), ps in groups.items()]


def _collate(seqs, pad_id, device):
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def _predict(model, seqs, pad_id, device, bs, amp):
    import torch
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), model.config.num_labels), np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            idx = order[s:s + bs]
            ids, mask = _collate([seqs[i] for i in idx], pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out[idx] = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
    return out


def train_and_predict(args, tok, tr_texts, tr_y, eval_sets, seed, K, class_w, save_dir=None):
    """Fine-tune one model; return logits for every named text list in eval_sets."""
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 300_000_000, f"model has {n_params / 1e6:.0f}M params (> 300M limit)"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    enc = tok(tr_texts, truncation=True, max_length=args.max_len)["input_ids"]
    n = len(enc)
    total = ((n + args.bs - 1) // args.bs) * args.epochs
    opt = torch.optim.AdamW(layerwise_groups(model, args.lr, args.lld, args.wd), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    GradScaler = torch.amp.GradScaler if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler
    scaler = GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") else GradScaler(enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device),
                                        label_smoothing=args.ls)
    rng = np.random.RandomState(seed)
    yt = torch.tensor(tr_y)
    model.train()
    step = 0
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        running = 0.0
        for s in range(0, n, args.bs):
            idx = perm[s:s + args.bs]
            ids, mask = _collate([enc[i] for i in idx], pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
        log(f"    epoch {ep + 1}/{args.epochs} loss {running / max(1, (n + args.bs - 1) // args.bs):.4f}")
    res = {}
    for name, texts in eval_sets.items():
        if len(texts) == 0:
            res[name] = np.zeros((0, K), np.float32)
            continue
        seqs = tok(texts, truncation=True, max_length=args.max_len)["input_ids"]
        res[name] = _predict(model, seqs, pad_id, device, 128, amp)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def tta(raw, alt, dev_mask):
    out = raw.copy()
    if dev_mask.any():
        out[dev_mask] = 0.5 * (raw[dev_mask] + alt)
    return log_softmax(out, axis=1)


def transformer_stage(args, D_tr, D_te, y, folds, K, pidx, py, rnd, cache_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    counts = np.bincount(y, minlength=K).astype(float)
    cw = (counts / counts.mean()) ** -0.5
    cw = cw / cw.mean()
    oof = np.zeros((len(y), K), np.float32)
    te = np.zeros((len(D_te.text), K), np.float32)
    for seed in args.seeds:
        for k, (tri, vai) in enumerate(folds):
            path = os.path.join(cache_dir, f"tf_r{rnd}_s{seed}_f{k}.npz")
            if os.path.exists(path):
                z = np.load(path)
                vl, tl = z["val"], z["test"]
                log(f"  transformer seed {seed} fold {k + 1}/{len(folds)}: loaded from cache")
            else:
                log(f"  transformer seed {seed} fold {k + 1}/{len(folds)}: training")
                texts, labels = gather(D_tr, tri, y, D_te, pidx, py, args.n_aug, use_norm=False)
                vdev, tdev = D_tr.dev[vai], D_te.dev
                ev = {"val": [D_tr.text[i] for i in vai],
                      "val_alt": [D_tr.norm[i] for i in vai[vdev]],
                      "te": D_te.text,
                      "te_alt": [D_te.norm[i] for i in np.where(tdev)[0]]}
                r = train_and_predict(args, tok, texts, labels, ev, seed * 1000 + k, K, cw)
                vl = tta(r["val"], r["val_alt"], vdev)
                tl = tta(r["te"], r["te_alt"], tdev)
                np.savez_compressed(path, val=vl, test=tl)
            oof[vai] += vl / len(args.seeds)
            te += tl / (len(args.seeds) * len(folds))
    return log_softmax(oof, axis=1), log_softmax(te, axis=1)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def write_submission(path, ids, pred, K):
    sub = pd.DataFrame({"example_id": np.asarray(ids), "target": np.asarray(pred).astype(int)})
    assert sub.example_id.is_unique and sub.target.between(0, K - 1).all()
    sub.to_csv(path, index=False)
    log(f"wrote {path} ({len(sub)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--skip_transformer", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lld", type=float, default=0.9, help="layer-wise LR decay")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--n_aug", type=int, default=2, help="noisy romanised copies per Devanagari row (transformer)")
    ap.add_argument("--tfidf_aug", type=int, default=1, help="noisy copies per Devanagari row (TF-IDF)")
    ap.add_argument("--svm_c", type=float, default=0.5)
    ap.add_argument("--group_thr", type=float, default=0.8, help="similarity above which rows share a fold")
    ap.add_argument("--pseudo_rounds", type=int, default=1)
    ap.add_argument("--pseudo_thr", type=float, default=0.9)
    ap.add_argument("--pseudo_max_per_class", type=int, default=150)
    ap.add_argument("--no_bias", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to test the code path")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    if args.fresh and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.smoke:
        df = df.sample(3000, random_state=0).reset_index(drop=True)
        test = test.head(200).reset_index(drop=True)
        args.folds, args.seeds, args.pseudo_thr = 2, [0], 0.0
    y = df["target"].values.astype(np.int64)
    K = 56
    assert y.max() < K
    log(f"train {len(df)} rows, test {len(test)} rows, {K} classes")

    t0 = time.time()
    D_tr = Data(df["query_text"], max(args.n_aug, args.tfidf_aug), seed=1)
    D_te = Data(test["query_text"], max(args.n_aug, args.tfidf_aug), seed=2)
    log(f"romanisation done in {time.time() - t0:.0f}s | Devanagari share: train {D_tr.dev.mean():.2f}, test {D_te.dev.mean():.2f}")
    log("example:", D_tr.text[int(np.where(D_tr.dev)[0][0])], "->", D_tr.norm[int(np.where(D_tr.dev)[0][0])])

    # ---- grouped folds + diagnostics comparing OOF similarity to test-vs-train similarity
    sim_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32).fit(
        D_tr.norm + D_te.norm)
    groups, Xtr = make_groups(D_tr.norm, args.group_thr, sim_vec)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    folds = [(np.array(a), np.array(b)) for a, b in sgkf.split(np.zeros(len(y)), y, groups)]
    s_test = nn_sim(sim_vec.transform(D_te.norm).tocsr(), Xtr)
    tri0, vai0 = folds[0]
    s_val = nn_sim(Xtr[vai0], Xtr[tri0])
    log(f"nearest-neighbour similarity to train pool | test: median {np.median(s_test):.2f}, >0.8: {(s_test > .8).mean():.1%}"
        f" | fold-0 validation: median {np.median(s_val):.2f}, >0.8: {(s_val > .8).mean():.1%}"
        "  (lower --group_thr if validation is still much closer to train than test is)")

    pidx, py = np.array([], dtype=int), np.array([], dtype=np.int64)
    summary = {}
    ens_oof = ens_te = None
    for rnd in range(args.pseudo_rounds + 1):
        log(f"===== round {rnd} (pseudo-labelled test rows in training: {len(pidx)}) =====")
        log("TF-IDF + LinearSVC")
        s_oof, s_te = tfidf_stage(D_tr, D_te, y, folds, K, args, pidx, py)
        T = fit_temperature(s_oof, y)
        svm_oof, svm_te = log_softmax(s_oof * T, axis=1), log_softmax(s_te * T, axis=1)
        summary[f"round{rnd}_svm"] = report("SVM", svm_oof, y, D_tr.dev, K)
        write_submission(os.path.join(args.out_dir, "submission_tfidf.csv"), test["example_id"], svm_te.argmax(1), K)

        if args.skip_transformer:
            ens_oof, ens_te, tf_te = svm_oof, svm_te, None
        else:
            log("Transformer:", args.model)
            tf_oof, tf_te = transformer_stage(args, D_tr, D_te, y, folds, K, pidx, py, rnd, cache_dir)
            summary[f"round{rnd}_transformer"] = report("Transformer", tf_oof, y, D_tr.dev, K)
            best_w, best_s = 0.0, -1
            for w in np.linspace(0, 1, 11):
                s = macro_f1(y, (w * tf_oof + (1 - w) * svm_oof).argmax(1), K)
                if s > best_s:
                    best_w, best_s = w, s
            log(f"blend weight on transformer: {best_w:.1f}")
            ens_oof = best_w * tf_oof + (1 - best_w) * svm_oof
            ens_te = best_w * tf_te + (1 - best_w) * svm_te
        summary[f"round{rnd}_ensemble"] = report("Ensemble", ens_oof, y, D_tr.dev, K)
        write_submission(os.path.join(args.out_dir, f"submission_round{rnd}.csv"), test["example_id"], ens_te.argmax(1), K)

        if rnd < args.pseudo_rounds:
            ens_te = log_softmax(ens_te, axis=1)
            conf, pred = np.exp(ens_te).max(1), ens_te.argmax(1)
            agree = np.ones(len(pred), bool) if tf_te is None else (tf_te.argmax(1) == svm_te.argmax(1))
            ok = (conf >= args.pseudo_thr) & agree
            keep = []
            for c in range(K):
                ci = np.where(ok & (pred == c))[0]
                keep.extend(ci[np.argsort(-conf[ci])][:args.pseudo_max_per_class])
            pidx = np.array(sorted(keep), dtype=int)
            py = pred[pidx].astype(np.int64)
            log(f"selected {len(pidx)} pseudo-labelled test rows (conf>={args.pseudo_thr}, models agree)")

    # ---- per-class bias tuning for macro-F1 (only kept if it helps on held-out groups)
    final = ens_te
    if not args.no_bias:
        gain = bias_heldout_gain(ens_oof, y, groups, K)
        log(f"per-class bias: held-out macro-F1 gain {gain:+.4f}")
        if gain > 0.001:
            b, best = tune_bias(ens_oof, y, K)
            log(f"applying bias (in-sample OOF macro-F1 {best:.4f}); class 55 bias {b[55]:+.2f}")
            final = ens_te + b
        else:
            log("bias not applied (no reliable held-out gain)")
    pred = final.argmax(1)
    log("class 55 share of final test predictions: %.1f%% (train share %.1f%%)" % (
        100 * (pred == 55).mean(), 100 * (y == 55).mean()))
    path = os.path.join(args.out_dir, "submission_smoke.csv" if args.smoke else "submission.csv")
    write_submission(path, test["example_id"], pred, K)
    if not args.smoke:
        assert len(pred) == 4912 or len(pred) == len(test)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("done. OOF summary:", json.dumps({k: round(v, 4) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
