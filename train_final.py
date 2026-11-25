#!/usr/bin/env python3
"""
Train the final, deployable model: ONE transformer + ONE TF-IDF SVM (<= 300M parameters in total).

What it does
  1. Holds out 10% of train (grouped by near-duplicate text, so the holdout is honest).
  2. Trains the SVM and the transformer on the other 90%.
  3. Uses the holdout to fit the SVM temperature and the blend weight, and reports macro-F1.
  4. Saves everything to <out_dir>/bundle  (loaded by nlu_infer.py / app.py).
  5. Optional --full_refit: retrains both on 100% of train with the settings chosen above.
  6. Writes <out_dir>/submission.csv by running test through the SAVED bundle, so the file you
     submit comes from exactly the model the UI serves.

Usage (GPU for the transformer; e.g. Colab T4, ~15-25 min for 5 epochs)
  python train_final.py --data_dir /path/to/csvs --out_dir final
  python train_final.py --data_dir /path/to/csvs --out_dir final --full_refit      # best final model
CPU only (SVM only, ~1 min):
  python train_final.py --data_dir /path/to/csvs --out_dir final_cpu --skip_transformer
"""
import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC

import nlu_pipeline as P
from nlu_infer import IntentClassifier

NO_IDX = np.array([], dtype=int)
NO_LAB = np.array([], dtype=np.int64)


def fit_svm(D_tr, D_te, idx, y, args):
    texts, labels = P.gather(D_tr, idx, y, D_te, NO_IDX, NO_LAB, args.tfidf_aug, use_norm=True)
    vec = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2),
                     TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True))
    X = vec.fit_transform(texts)
    model = LinearSVC(C=args.svm_c, class_weight="balanced").fit(X, labels)
    return vec, model


def fit_transformer(args, tok, D_tr, D_te, idx, y, K, eval_sets, save_dir):
    texts, labels = P.gather(D_tr, idx, y, D_te, NO_IDX, NO_LAB, args.n_aug, use_norm=False)
    counts = np.bincount(labels, minlength=K).astype(float)
    cw = (counts / counts.mean()) ** -0.5
    cw = cw / cw.mean()
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    return P.train_and_predict(args, tok, texts, labels, eval_sets, args.seed, K, cw, save_dir=save_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="final")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--skip_transformer", action="store_true")
    ap.add_argument("--full_refit", action="store_true", help="retrain on 100%% of train after tuning on the holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lld", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--n_aug", type=int, default=2)
    ap.add_argument("--tfidf_aug", type=int, default=1)
    ap.add_argument("--svm_c", type=float, default=0.5)
    ap.add_argument("--group_thr", type=float, default=0.8)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to test the code path")
    args = ap.parse_args()

    bundle = os.path.join(args.out_dir, "bundle")
    os.makedirs(bundle, exist_ok=True)
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.smoke:
        df = df.sample(3000, random_state=0).reset_index(drop=True)
        test = test.head(200).reset_index(drop=True)
    y = df["target"].values.astype(np.int64)
    K = 56
    P.set_seed(args.seed)

    D_tr = P.Data(df["query_text"], max(args.n_aug, args.tfidf_aug), seed=1)
    D_te = P.Data(test["query_text"], max(args.n_aug, args.tfidf_aug), seed=2)

    # ---- grouped 90/10 split
    sim_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32).fit(D_tr.norm)
    groups, _ = P.make_groups(D_tr.norm, args.group_thr, sim_vec)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    tr_idx, ho_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    P.log(f"train {len(tr_idx)} rows / holdout {len(ho_idx)} rows")
    yho = y[ho_idx]
    dev_ho = D_tr.dev[ho_idx]

    # ---- SVM on 90%, temperature on holdout
    P.log("fitting SVM")
    vec, svm = fit_svm(D_tr, D_te, tr_idx, y, args)
    s_ho = svm.decision_function(vec.transform([D_tr.norm[i] for i in ho_idx]))
    T = P.fit_temperature(s_ho, yho)
    svm_lp = log_softmax(s_ho * T, axis=1)
    P.report("SVM (holdout)", svm_lp, yho, dev_ho, K)
    blend_w, tf_params, tf_dir = 0.0, 0, os.path.join(bundle, "transformer")

    # ---- transformer on 90%, blend weight on holdout
    if not args.skip_transformer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        P.log("fine-tuning transformer:", args.model)
        ev = {"ho": [D_tr.text[i] for i in ho_idx], "ho_alt": [D_tr.norm[i] for i in ho_idx[dev_ho]]}
        r = fit_transformer(args, tok, D_tr, D_te, tr_idx, y, K, ev, tf_dir)
        tf_lp = P.tta(r["ho"], r["ho_alt"], dev_ho)
        P.report("Transformer (holdout)", tf_lp, yho, dev_ho, K)
        scores = {w: P.macro_f1(yho, (w * tf_lp + (1 - w) * svm_lp).argmax(1), K) for w in np.linspace(0, 1, 11)}
        blend_w = float(max(scores, key=scores.get))
        P.log(f"blend weight on transformer: {blend_w:.1f} -> holdout macro-F1 {scores[blend_w]:.4f}")

    # ---- optional refit on all data
    if args.full_refit:
        P.log("refitting on 100% of train")
        vec, svm = fit_svm(D_tr, D_te, np.arange(len(y)), y, args)
        if not args.skip_transformer:
            fit_transformer(args, tok, D_tr, D_te, np.arange(len(y)), y, K, {}, tf_dir)

    # ---- parameter budget
    svm_params = int(svm.coef_.size + svm.intercept_.size)
    if not args.skip_transformer:
        from transformers import AutoModelForSequenceClassification
        tf_params = sum(p.numel() for p in AutoModelForSequenceClassification.from_pretrained(tf_dir).parameters())
    total = tf_params + svm_params
    P.log(f"parameters: transformer {tf_params / 1e6:.1f}M + SVM {svm_params / 1e6:.1f}M = {total / 1e6:.1f}M (limit 300M)")
    assert total <= 300_000_000, "over the 300M parameter budget"

    # ---- save bundle
    joblib.dump({"vec": vec, "model": svm}, os.path.join(bundle, "svm.joblib"), compress=3)
    meta = {"K": K, "blend_w": blend_w, "svm_T": float(T), "max_len": args.max_len, "base_model": args.model,
            "params_transformer": int(tf_params), "params_svm": svm_params, "params_total": int(total),
            "full_refit": bool(args.full_refit), "holdout_note": "blend weight and temperature tuned on 10% grouped holdout"}
    with open(os.path.join(bundle, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- verify the saved bundle end to end, then write the submission from it
    clf = IntentClassifier(bundle)
    pred_ho = clf.predict_proba([D_tr.text[i] for i in ho_idx]).argmax(1)
    P.log(f"saved bundle reloaded: holdout macro-F1 {P.macro_f1(yho, pred_ho, K):.4f}"
          + ("  (NOTE: holdout rows were in training because of --full_refit, so this is optimistic)" if args.full_refit else ""))
    pred = clf.predict_proba(test["query_text"].tolist()).argmax(1)
    P.write_submission(os.path.join(args.out_dir, "submission_smoke.csv" if args.smoke else "submission.csv"),
                       test["example_id"], pred, K)
    P.log("class 55 share of test predictions: %.1f%%" % (100 * (pred == 55).mean()))
    P.log(f"done. bundle at {bundle}")


if __name__ == "__main__":
    main()
