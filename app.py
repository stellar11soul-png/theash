#!/usr/bin/env python3
"""
Web UI for the Hindi / Hinglish intent classifier.

  pip install gradio
  python app.py --bundle final/bundle                 # http://127.0.0.1:7860
  python app.py --bundle final/bundle --share         # temporary public link (e.g. from Colab)

Tab 1: type a query (Devanagari or Roman Hinglish) -> predicted intent, confidence, top-5.
Tab 2: upload a CSV with a `query_text` column -> download predictions
       (if the CSV also has `example_id`, the download is in the competition submission format).
Optional: --labels labels.json   ({"0": "set_alarm", ...}) to show readable intent names.
"""
import argparse
import os
import tempfile

import gradio as gr
import pandas as pd

from nlu_infer import IntentClassifier

GRADIO6 = int(gr.__version__.split(".")[0]) >= 6

EXAMPLES = [
    "कृपया रसोईघर की बत्तियां बंद करें",
    "kripaya rasoeeghar ki battiyaan band kar den",
    "aaj ka mausam kaisa hai",
    "मुझे अगला गाना सुनाओ",
    "samay kya hua hai",
    "mujhe ek chess game ke rules batao",
]


def build_ui(clf):
    def classify(text, threshold):
        text = (text or "").strip()
        if not text:
            return {}, "Type a query above."
        r = clf.predict([text], topk=5, fallback_threshold=threshold)[0]
        top = {clf.name(i): p for i, p in r["top"]}
        verdict = f"### {clf.name(r['pred'])}\nConfidence: **{r['confidence']:.1%}**"
        if r["fell_back"]:
            verdict += f"  \nBelow the {threshold:.0%} threshold, so it was routed to the fallback class."
        verdict += f"\n\nInput script: {r['script']}  \nNormalised text seen by the model: `{r['normalized']}`"
        return top, verdict

    def classify_file(file, threshold):
        if file is None:
            return None, None
        path = file if isinstance(file, str) else file.name
        df = pd.read_csv(path)
        if "query_text" not in df.columns:
            raise gr.Error("The CSV needs a `query_text` column.")
        res = clf.predict(df["query_text"].astype(str).tolist(), topk=1, fallback_threshold=threshold)
        out = pd.DataFrame({
            "query_text": df["query_text"],
            "predicted_class": [r["pred"] for r in res],
            "intent": [clf.name(r["pred"]) for r in res],
            "confidence": [round(r["confidence"], 4) for r in res],
        })
        ids = df["example_id"] if "example_id" in df.columns else pd.Series(range(len(df)))
        tmp = tempfile.mkdtemp()
        sub_path = os.path.join(tmp, "submission.csv")
        pd.DataFrame({"example_id": ids, "target": out["predicted_class"]}).to_csv(sub_path, index=False)
        return out.head(200), sub_path

    blocks_kw = {} if GRADIO6 else {"theme": gr.themes.Soft()}   # Gradio 6 moved theme to launch()
    with gr.Blocks(title="Hindi / Hinglish Intent Classifier", **blocks_kw) as demo:
        gr.Markdown("# Hindi / Hinglish intent classifier\n"
                    "56 intents including class 55 (Out-of-Scope / Fallback). Works with Devanagari and Roman Hinglish.")
        threshold = gr.Slider(0, 0.95, value=0.0, step=0.05,
                              label="Fallback threshold: send to Out-of-Scope when confidence is below this (0 = off)")
        with gr.Tab("Single query"):
            inp = gr.Textbox(label="Query", placeholder="e.g. kripaya rasoeeghar ki battiyaan band kar den", lines=2)
            btn = gr.Button("Classify", variant="primary")
            with gr.Row():
                verdict = gr.Markdown()
                label = gr.Label(num_top_classes=5, label="Top intents")
            gr.Examples(EXAMPLES, inputs=inp)
            btn.click(classify, [inp, threshold], [label, verdict], api_name="classify")
            inp.submit(classify, [inp, threshold], [label, verdict])
        with gr.Tab("Batch CSV"):
            gr.Markdown("Upload a CSV with a `query_text` column (add `example_id` to get a ready-to-submit file).")
            up = gr.File(label="CSV file", file_types=[".csv"])
            run = gr.Button("Predict", variant="primary")
            preview = gr.Dataframe(label="Preview (first 200 rows)")
            download = gr.File(label="Download submission.csv")
            run.click(classify_file, [up, threshold], [preview, download], api_name="batch")
        meta = clf.meta
        gr.Markdown(f"<small>Model: {'transformer + ' if clf.tf is not None else ''}TF-IDF SVM | "
                    f"{meta.get('params_total', 0) / 1e6:.0f}M parameters (limit 300M)</small>")
    return demo


def main():
    ap = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ap.add_argument(
    "--bundle",
    default=os.environ.get(
        "BUNDLE_DIR",
        os.path.join(BASE_DIR, "final", "bundle")
    )
)
  ap.add_argument(
    "--labels",
    default=os.environ.get(
        "LABELS_JSON",
        os.path.join(BASE_DIR, "labels.json")
    )
)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    clf = IntentClassifier(args.bundle, args.labels)
    launch_kw = {"theme": gr.themes.Soft()} if GRADIO6 else {}
    build_ui(clf).launch(server_name=args.host, server_port=args.port, share=args.share, **launch_kw)


if __name__ == "__main__":
    main()
