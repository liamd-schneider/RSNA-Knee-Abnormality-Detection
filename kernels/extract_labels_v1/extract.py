"""RSNA Knee Abnormality Detection -- report-label extraction (v1).

The core data problem for this competition: only 58 of 4,407 training
studies have expert labels; the other 4,349 only have the free-text
radiology report (in whichever of roughly a dozen languages the reporting
site used). This script uses a small open-weight instruction-following LLM
to read a report and decide, for each of the 12 target findings, whether
it's present.

Two modes, controlled by KNEE_MODE:
  - "validate" (default, safe): run ONLY on the 58 gold-labeled studies and
    score the extraction against their real labels -- this must be checked
    BEFORE trusting the extractor on the other 4,349, exactly like the
    process documented in notebooks/reference-notes.md (their prompt was
    revised against the same 58 gold studies before being trusted at scale).
  - "full": run on the remaining ~4,349 unlabeled studies and save the
    extracted labels. Only meant to be run after "validate" shows the
    extractor is actually decent.

Local dry run (no GPU/model download, just exercises the JSON parser):
    KNEE_SKIP_TORCH_INSTALL=1 KNEE_TEST_PARSE_ONLY=1 python extract.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------- torch bootstrap
# Same fix as kernels/train_v3: Kaggle's allocated P100 (compute capability
# sm_60) isn't supported by the pre-installed torch build, so install one
# that still ships sm_60 kernels before torch is ever imported.
if not os.environ.get("KNEE_SKIP_TORCH_INSTALL"):
    for candidate in ["torch==2.3.1", "torch==2.1.2"]:
        try:
            print(f"[setup] installing {candidate} (cu121, sm_60/P100 support)...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", candidate,
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True, timeout=600,
            )
            break
        except Exception as e:
            print(f"[setup] {candidate} install failed ({e}), trying next candidate", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "transformers>=4.45,<4.50", "accelerate>=0.33", "safetensors"],
            check=True, timeout=600,
        )
    except Exception as e:
        print(f"[setup] transformers/accelerate install failed: {e}", flush=True)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

MODEL_NAME = os.environ.get("KNEE_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODE = os.environ.get("KNEE_MODE", "validate")
BATCH_SIZE = int(os.environ.get("KNEE_BATCH_SIZE", "8"))
MAX_NEW_TOKENS = int(os.environ.get("KNEE_MAX_NEW_TOKENS", "200"))
MAX_REPORT_CHARS = 3000
OUT_DIR = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")

SYSTEM_PROMPT = f"""You are assisting with structured data extraction from knee MRI radiology reports for a research dataset. Reports may be written in English, Spanish, French, or other languages -- read the report in its original language, but always answer in English using the exact JSON schema below.

For each of these 12 findings, decide whether the report indicates it is PRESENT (1) or not mentioned / explicitly absent (0):

- ACL: any anterior cruciate ligament tear, sprain, or rupture (partial or complete).
- MCL: any medial collateral ligament tear or sprain (partial or complete).
- Medial Meniscus: any tear of the medial meniscus.
- Lateral Meniscus: any tear of the lateral meniscus.
- Medial OA: osteoarthritis / degenerative changes (joint space narrowing, osteophytes, cartilage loss) in the MEDIAL tibiofemoral compartment.
- Lateral OA: osteoarthritis / degenerative changes in the LATERAL tibiofemoral compartment.
- PF OA: osteoarthritis / degenerative changes in the patellofemoral compartment.
- Effusion: joint effusion / fluid in the joint. Count any explicitly mentioned effusion, including "trace" or "small" amounts, as present. If effusion is not mentioned at all, mark absent.
- Synovitis: synovitis, synovial thickening/inflammation, Hoffa fat pad impingement, plica syndrome, or friction/impingement syndrome findings.
- Baker's: Baker's cyst / popliteal cyst.
- Contusion: acute bone contusion / bone bruise with a described traumatic mechanism or acute marrow edema pattern. Do NOT count ordinary chronic degenerative or osteoarthritic subchondral marrow edema as contusion.
- Fracture: any fracture, including acute, avulsion, or insufficiency/stress fractures.

If a finding is simply not discussed in the report, treat it as absent (0) -- do not guess.

Respond with ONLY a single JSON object, no other text, no explanation, using exactly these keys:
{{"ACL": 0, "MCL": 0, "Medial Meniscus": 0, "Lateral Meniscus": 0, "Medial OA": 0, "Lateral OA": 0, "PF OA": 0, "Effusion": 0, "Synovitis": 0, "Baker's": 0, "Contusion": 0, "Fracture": 0}}"""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no_json_found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "json_decode_error"
    if not isinstance(obj, dict):
        return None, "not_a_dict"
    result = {}
    for k in LABELS:
        v = obj.get(k, 0)
        try:
            result[k] = int(bool(int(v))) if not isinstance(v, bool) else int(v)
        except (TypeError, ValueError):
            result[k] = 0
    return result, None


def _self_test_parser():
    cases = [
        ('{"ACL": 1, "MCL": 0}', {"ACL": 1, "MCL": 0}),
        ('Sure, here is the JSON:\n{"ACL": 0, "Effusion": 1}', {"ACL": 0, "Effusion": 1}),
        ("not json at all", None),
    ]
    for text, expected_subset in cases:
        result, err = parse_json(text)
        if expected_subset is None:
            assert result is None, f"expected parse failure for {text!r}, got {result}"
        else:
            for k, v in expected_subset.items():
                assert result[k] == v, f"{text!r}: expected {k}={v}, got {result[k]}"
    print("[selftest] parse_json OK", flush=True)


def find_input_dir():
    env = os.environ.get("KNEE_INPUT_DIR")
    if env and os.path.exists(os.path.join(env, "train.csv")):
        return env
    candidates = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise RuntimeError("could not locate train.csv under /kaggle/input")


def main():
    if os.environ.get("KNEE_TEST_PARSE_ONLY"):
        _self_test_parser()
        return

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    input_dir = find_input_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] torch={torch.__version__} device={device} model={MODEL_NAME} "
          f"mode={MODE} batch_size={BATCH_SIZE}", flush=True)

    # train.csv is latin-1 encoded, not UTF-8 -- reading with the pandas
    # default silently mangles every accented character in the (multilingual)
    # Report column instead of raising an error.
    train = pd.read_csv(os.path.join(input_dir, "train.csv"), encoding="latin-1")
    gold = train.dropna(subset=LABELS, how="all").reset_index(drop=True)
    print(f"[data] {len(train)} studies total, {len(gold)} gold-labeled", flush=True)

    if MODE == "validate":
        target = gold
    elif MODE == "full":
        target = train[~train["StudyInstanceUID"].isin(gold["StudyInstanceUID"])].reset_index(drop=True)
        target = target.dropna(subset=["Report"]).reset_index(drop=True)
    else:
        raise ValueError(f"unknown KNEE_MODE={MODE!r}")
    print(f"[data] extracting for {len(target)} studies ({MODE} mode)", flush=True)

    print("[setup] loading model (first run downloads weights)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"[setup] model loaded, elapsed={time.time()-t0:.0f}s", flush=True)

    def run_batch(reports):
        texts = []
        for r in reports:
            r = (r or "")[:MAX_REPORT_CHARS]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Report:\n"""\n{r}\n"""\n\nJSON:'},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        results = []
        for i in range(len(reports)):
            gen = out[i][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            parsed, err = parse_json(text)
            results.append((parsed, err, text[:300]))
        return results

    rows = []
    n_parse_fail = 0
    reports_list = target["Report"].tolist()
    uids = target["StudyInstanceUID"].tolist()
    for start in range(0, len(reports_list), BATCH_SIZE):
        batch_reports = reports_list[start:start + BATCH_SIZE]
        batch_uids = uids[start:start + BATCH_SIZE]
        batch_results = run_batch(batch_reports)
        for uid, (parsed, err, raw) in zip(batch_uids, batch_results):
            if parsed is None:
                n_parse_fail += 1
                parsed = {k: 0 for k in LABELS}
            row = {"StudyInstanceUID": uid, "parse_failed": err is not None}
            row.update(parsed)
            rows.append(row)
        done = min(start + BATCH_SIZE, len(reports_list))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"[extract {done}/{len(reports_list)}] elapsed={elapsed:.0f}s "
              f"rate={rate:.2f} studies/s parse_failures={n_parse_fail}", flush=True)

    pred_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)

    if MODE == "validate":
        merged = gold.merge(pred_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
        per_label = {}
        for lab in LABELS:
            y_true = merged[f"{lab}_true"].astype(int)
            y_pred = merged[f"{lab}_pred"].astype(int)
            per_label[lab] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "positive_rate_true": float(y_true.mean()),
                "positive_rate_pred": float(y_pred.mean()),
            }
        mean_acc = float(np.mean([v["accuracy"] for v in per_label.values()]))
        mean_f1 = float(np.mean([v["f1"] for v in per_label.values()]))
        report = {
            "model": MODEL_NAME, "n_gold": len(merged), "n_parse_failed": n_parse_fail,
            "mean_accuracy": mean_acc, "mean_f1": mean_f1, "per_label": per_label,
        }
        with open(os.path.join(OUT_DIR, "validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[done] mean_accuracy={mean_acc:.4f} mean_f1={mean_f1:.4f} "
              f"parse_failures={n_parse_fail}/{len(merged)}", flush=True)
        print(f"[done] per_label: {json.dumps(per_label, indent=2)}", flush=True)
    else:
        out_path = os.path.join(OUT_DIR, "report_labels_extracted.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"[done] wrote {out_path} ({len(pred_df)} studies, "
              f"{n_parse_fail} parse failures defaulted to all-0)", flush=True)

    print(f"[done] total wall time {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
