"""
Inference wrapper for the saved model bundle (used by train_final.py and app.py).

Bundle layout (created by train_final.py):
  bundle/meta.json        class count, blend weight, SVM temperature, parameter counts, holdout score
  bundle/svm.joblib       TF-IDF vectorizer + LinearSVC
  bundle/transformer/     fine-tuned MuRIL / XLM-R (optional; absent when trained with --skip_transformer)

Optional labels file (labels.json) maps class id -> readable intent name, e.g. {"0": "set_alarm", "55": "out_of_scope"}.
The dataset only gives numeric classes, so without it the UI shows "Intent <id>".
"""
import json
import os

import joblib
import numpy as np
from scipy.special import log_softmax, softmax

from nlu_pipeline import is_dev, norm, rom

DEFAULT_LABELS = {"55": "Out-of-Scope / Fallback"}


class IntentClassifier:
    def __init__(self, bundle_dir, labels_path=None):
        with open(os.path.join(bundle_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.K = self.meta["K"]
        self.blend_w = self.meta.get("blend_w", 0.0)
        self.T = self.meta["svm_T"]
        svm = joblib.load(os.path.join(bundle_dir, "svm.joblib"))
        self.vec, self.svm = svm["vec"], svm["model"]

        self.tf = None
        tdir = os.path.join(bundle_dir, "transformer")
        if os.path.isdir(tdir):
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(tdir)
            self.tf = AutoModelForSequenceClassification.from_pretrained(tdir).to(self.device).eval()
            self.max_len = self.meta.get("max_len", 64)

        self.labels = dict(DEFAULT_LABELS)
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                self.labels.update({str(k): v for k, v in json.load(f).items()})

    # ---- helpers -----------------------------------------------------------------
    def name(self, i):
        n = self.labels.get(str(i))
        return f"{n} (class {i})" if n else f"Intent {i}"

    @staticmethod
    def normalize(text):
        return rom(text) if is_dev(text) else norm(text)

    def _tf_logits(self, texts, bs=64):
        torch = self.torch
        out = np.zeros((len(texts), self.K), np.float32)
        order = np.argsort([len(t) for t in texts])
        amp = self.device.type == "cuda"
        with torch.no_grad():
            for s in range(0, len(texts), bs):
                idx = order[s:s + bs]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    logits = self.tf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                out[idx] = logits.float().cpu().numpy()
        return out

    # ---- public API --------------------------------------------------------------
    def predict_proba(self, texts):
        texts = [str(t) for t in texts]
        normed = [self.normalize(t) for t in texts]
        svm_lp = log_softmax(self.svm.decision_function(self.vec.transform(normed)) * self.T, axis=1)
        if self.tf is None:
            return softmax(svm_lp, axis=1)
        dev = np.array([is_dev(t) for t in texts])
        raw = self._tf_logits(texts)
        if dev.any():                       # test-time augmentation: original + romanised for Devanagari input
            alt = self._tf_logits([normed[i] for i in np.where(dev)[0]])
            raw[dev] = 0.5 * (raw[dev] + alt)
        tf_lp = log_softmax(raw, axis=1)
        return softmax(self.blend_w * tf_lp + (1 - self.blend_w) * svm_lp, axis=1)

    def predict(self, texts, topk=5, fallback_threshold=0.0, fallback_class=55):
        """Return one dict per text with prediction, confidence and top-k alternatives."""
        texts = [str(t) for t in texts]
        P = self.predict_proba(texts)
        res = []
        for t, p in zip(texts, P):
            order = np.argsort(-p)
            pred = int(order[0])
            fell_back = bool(p[pred] < fallback_threshold and pred != fallback_class)
            res.append({
                "text": t,
                "script": "Devanagari" if is_dev(t) else "Roman",
                "normalized": self.normalize(t),
                "pred": fallback_class if fell_back else pred,
                "confidence": float(p[pred]),
                "fell_back": fell_back,
                "top": [(int(i), float(p[i])) for i in order[:topk]],
            })
        return res
