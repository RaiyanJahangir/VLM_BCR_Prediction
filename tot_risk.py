#!/usr/bin/env python3
import argparse
import os
import json
import re
import logging
import concurrent.futures
import subprocess
from pathlib import Path

from langchain_ollama import OllamaLLM as Ollama

# ── Paths (adjust if needed) ────────────────────────────────────────────────────
REPORT_DIR = ".../embed/per_patient_json"  # The directory where the json reports are
OUTPUT_DIR = Path("result")  # the directory in which you want to save outputs

# ── Prompt ──────────────────────────────────────────────────────────────────────
PROMPT = r"""
You are a board-certified breast radiologist and a biostatistician specializing in longitudinal breast cancer risk modeling. You are given, for ONE patient at a time, a sequence of mammogram exams (oldest to newest) and their corresponding clinical history, all in a single JSON object.

ASSUME:
- The exams array is ordered from oldest to newest.
- One or more views or clinical fields may be missing or null.
- You will also receive the corresponding images referenced in "images" for each exam.

IMPORTANT: You must reason using a Tree-of-Thought process internally (exploring multiple candidate explanations and risk curves), but your FINAL output must ONLY be the JSON object described below, with NO intermediate reasoning, NO explanations, and NO extra text.

------------------------------------------------
INTERNAL TREE-OF-THOUGHT REASONING (NOT OUTPUT)
------------------------------------------------

[TO ROOT] Understand the input
- Parse the patient JSON:
  - Record patient_id and demographics (age, family history, genetic risk, other risk factors).
  - Record each exam's date, age, available image views, and clinical notes.
- Note missing views or fields but do not let them block reasoning; treat them as uncertainty.

[LEVEL 1 BRANCHING] Generate candidate global interpretations of the longitudinal pattern

From the entire sequence of exams and clinical data, generate at least 3 candidate ``global stories'' (paths) for the patient's trajectory. For example:

- Path A (Low-risk story):
  - All exams appear normal or benign.
  - No suspicious evolution, no strong risk factors.
  - Represents a ``consistently low risk'' interpretation.

- Path B (Intermediate-risk story):
  - Some probably benign findings (BIRADS 3-like) that are stable or slowly evolving.
  - Moderate background risk ( age, mild family history).
  - Represents a ``moderate, gradually increasing risk'' interpretation.

- Path C (High-risk story):
  - The latest exam suggests a NEW or CLEARLY PROGRESSIVE suspicious lesion (BIRADS 4 or 5-like).
  - And/or strong clinical risk factors ( BRCA+, strong family history, prior cancer).
  - Represents an ``elevated near-term risk'' interpretation.

If one of these does not fit the actual data, adjust it to match better the observed pattern (for example, all paths might be low or moderate; or add a fourth path if needed). Each path must be:
- A plausible qualitative explanation of the longitudinal exams.
- Internally consistent with imaging and clinical data.

[LEVEL 2 BRANCHING] For EACH candidate path, derive a plausible 1 to 8-year risk curve

For each path (A, B, C, ...), construct a candidate 8-year cumulative risk trajectory:

For y in {1, 2, 3, 4, 5, 6, 7, 8}:
- Candidate risk_y = P(developing breast cancer within y years from now),
  expressed as a probability in [0,1].

The trajectory MUST satisfy:
- 0 < risk_1y < risk_2y < ... < risk_8y < 1 (monotonic cumulative risk).

Use the following heuristics:

- If the path describes consistently regular/benign exams (BIRADS 1 to 2 conceptually) and no strong risk factors:
  - Keep all risks low and slowly increasing (by 8 years often < ~0.10).

- If the path describes probably benign but stable findings (BIRADS 3 conceptually):
  - Use modest but gently rising risks (higher than entirely benign but still smooth and not extreme).

- If the path describes a new or clearly progressive suspicious lesion on the latest exam (BIRADS 4 to 5 conceptually):
  - Increase near-term risks (1 to 3 years) and also later horizons.
  - Ensure that early years show substantially higher risk than the entirely benign scenarios.

- If the path includes substantial clinical background risk (BRCA carrier, strong family history, prior cancer):
  - Shift the entire curve upward, but keep it monotone and consistent with the narrative.

For each path:
- Check that all risk_y are within [0,1].
- Check that risk_1y < risk_2y < ... < risk_8y.

[LEVEL 3 BRANCHING] For EACH candidate path, derive time-to-event estimates

For each candidate path with its candidate risk curve:

- Propose:
  - candidate expected years: a buoyant float, your best estimate of when cancer would most likely develop, IF it produces, measured in years from now.
  - candidate_bucket: one of:
      "<1 year",
      "1 to 3 years",
      "3 to 5 years",
      "5 to 8 years",
      ">=8 years or uncertain"

Heuristics for each candidate path:

- If the path is low-risk with benign exams and a low-risk curve:
  - Choose a longer bucket like "5 to 8 years" or ">=8 years or uncertain".
  - Candidate expected years should be relatively large ( 6 to 10).

- If the path is high-risk or strongly suspicious on the latest exam:
  - Choose a shorter bucket like "<1 year" or "1 to 3 years".
  - Candidate expected years should be close to the near-term horizon.

- Ensure consistency:
  - If candidate_bucket is very short ( "<1 year"), the early risk values (risk_1y, risk_2y) must be noticeably higher than in low-risk paths.

[LEVEL 4 SELECTION] Evaluate and choose the best path

Now compare all candidate paths (A, B, C, ...):

For each path, internally evaluate:
- How closely it matches the actual imaging findings over time.
- How well does it match the clinical risk factors?
- How plausible is the risk curve and time-to-event estimate, given your radiologic and statistical intuition?

Choose the **single best path** that:
- Is most consistent with the actual patient trajectory.
- Has a coherent risk curve and time-to-event estimate.
- Respects all constraints (probability range, monotonicity, plausible magnitudes).

Discard the other paths. Use ONLY the selected path for your final output.

[FINAL INTERNAL CHECK]
For the chosen path:
- Confirm:
  - All risk_y are in [0,1].
  - risk_1y < risk_2y < ... < risk_8y.
  - expected_years > 0.
  - bucket is one of the allowed strings.

You are NOT a calibrated medical risk model. These outputs are approximate, for research use only.

DO NOT output any of these reasoning steps. They are for your internal Tree-of-Thought process only.

--------------------------------------
FINAL OUTPUT FORMAT (ONLY THIS JSON)
--------------------------------------

Return ONLY a single JSON object with NO extra commentary, in the following format:

{
  "patient_id": "<copy from input>",
  "risk_estimates": {
    "risk_1y": <float in [0,1]>,
    "risk_2y": <float in [0,1]>,
    "risk_3y": <float in [0,1]>,
    "risk_4y": <float in [0,1]>,
    "risk_5y": <float in [0,1]>,
    "risk_6y": <float in [0,1]>,
    "risk_7y": <float in [0,1]>,
    "risk_8y": <float in [0,1]>
  },
  "time_to_event": {
    "expected_years": <float > 0>,
    "bucket": "<one of: "<1 year", "1 to 3 years", "3 to 5 years", "5 to 8 years", ">=8 years or uncertain">"
  }
}

CONSTRAINTS:
- Do NOT hallucinate impossible probabilities (all must be between 0 and 1).
- Enforce monotonicity: risk_1y < risk_2y < ... < risk_8y.
- Do NOT output any treatment recommendations or clinical management advice.
- Do NOT output anything outside of the JSON object.
""".strip()

# ── Risk keys for sanity checks ────────────────────────────────────────────────
RISK_KEYS = [f"risk_{i}y" for i in range(1, 9)]

# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_ollama_model(model_name: str):
    """
    Ensure the Ollama model is available locally.
    If not, attempt to `ollama pull <model_name>`.
    """
    try:
        # Check if model exists
        show = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
        )
        if show.returncode != 0:
            print(f"📥 Model '{model_name}' not found locally. Pulling with `ollama pull {model_name}`...")
            pull = subprocess.run(["ollama", "pull", model_name])
            if pull.returncode != 0:
                raise RuntimeError(f"Failed to pull Ollama model '{model_name}'.")
        else:
            print(f"✅ Ollama model '{model_name}' is available.")
    except FileNotFoundError:
        raise RuntimeError("`ollama` CLI not found. Please install Ollama and ensure it is on your PATH.")


def list_json_reports(root: str):
    all_files = sorted([
        fn for fn in os.listdir(root)
        if fn.lower().endswith(".json")
    ])
    return all_files


def call_ollama(model: str, system_prompt: str, patient_json_str: str, timeout_seconds: int = 600):
    """
    Call Ollama with a timeout, passing the system prompt plus the patient JSON.
    Returns the raw text response or None on timeout.
    """
    def _query():
        # We just concatenate system prompt + patient JSON in one message,
        # since langchain_ollama.OllamaLLM uses a single string interface.
        query = (
            system_prompt
            + "\n\nHere is the JSON object for ONE patient:\n\n"
            + patient_json_str
        )
        ollama = Ollama(model=model, temperature=0)
        logging.getLogger().setLevel(logging.ERROR)
        return ollama.invoke(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_query)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return None


def extract_json_block(text: str) -> str:
    """
    Extract the first {...} block from the text and clean trailing commas.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return ""
    s = text[start : end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
    return s


def enforce_risk_monotonicity(risks: dict, eps: float = 1e-3) -> dict:
    """
    Make sure:
      0 <= risk_1y < risk_2y < ... < risk_8y <= 1
    Fill missing keys by small increments.
    """
    cleaned = {}
    prev = 0.0
    n = len(RISK_KEYS)
    for i, key in enumerate(RISK_KEYS):
        raw = risks.get(key, None)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = prev + eps

        # clamp
        val = max(0.0, min(1.0, val))

        # strictly increasing
        if val <= prev + eps:
            val = prev + eps

        remaining = n - i - 1
        if remaining > 0:
            upper_bound = 1.0 - remaining * eps
            if val > upper_bound:
                val = upper_bound
        else:
            val = min(val, 1.0 - eps)

        cleaned[key] = float(val)
        prev = val
    return cleaned


def sanitize_time_to_event(tte: dict) -> dict:
    """
    Ensure expected_years > 0 and bucket is valid.
    """
    allowed = {
        "< 1 year",
        "1 to 3 years",
        "3 to 5 years",
        "5 to 8 years",
        "> 8 years_or_uncertain",
    }
    expected = tte.get("expected_years", 5.0)
    try:
        expected = float(expected)
    except (TypeError, ValueError):
        expected = 5.0
    if expected <= 0:
        expected = 5.0

    bucket = tte.get("bucket", "> 8 years_or_uncertain")
    if bucket not in allowed:
        if expected < 1:
            bucket = "< 1 year"
        elif expected < 3:
            bucket = "1 to 3 years"
        elif expected < 5:
            bucket = "3 to 5 years"
        elif expected < 8:
            bucket = "5 to 8 years"
        else:
            bucket = "> 8 years_or_uncertain"

    return {
        "expected_years": float(expected),
        "bucket": bucket,
    }


def normalize_result(text: str, fallback_patient_id: str):
    """
    Parse the model output into the required JSON structure:
    {
      "patient_id": "...",
      "risk_estimates": {...},
      "time_to_event": {...}
    }
    """
    if not text or not text.strip():
        # pure fallback
        return {
            "patient_id": fallback_patient_id,
            "risk_estimates": {k: 0.0 for k in RISK_KEYS},
            "time_to_event": {
                "expected_years": 5.0,
                "bucket": "> 8 years_or_uncertain",
            },
        }, "", False

    block = extract_json_block(text)
    if not block:
        return {
            "patient_id": fallback_patient_id,
            "risk_estimates": {k: 0.0 for k in RISK_KEYS},
            "time_to_event": {
                "expected_years": 5.0,
                "bucket": "> 8 years_or_uncertain",
            },
        }, "", False

    try:
        data = json.loads(block)
    except Exception:
        return {
            "patient_id": fallback_patient_id,
            "risk_estimates": {k: 0.0 for k in RISK_KEYS},
            "time_to_event": {
                "expected_years": 5.0,
                "bucket": "> 8 years_or_uncertain",
            },
        }, block, False

    # patient_id
    patient_id = data.get("patient_id", fallback_patient_id)

    # risk_estimates
    risk_dict = data.get("risk_estimates", {}) or {}
    risk_clean = enforce_risk_monotonicity(risk_dict)

    # time_to_event
    tte_dict = data.get("time_to_event", {}) or {}
    tte_clean = sanitize_time_to_event(tte_dict)

    out = {
        "patient_id": patient_id,
        "risk_estimates": risk_clean,
        "time_to_event": tte_clean,
    }
    return out, block, True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal breast cancer risk estimation via Ollama LLM from JSON reports."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name (anas/video-llava:test, qwen2.5vl, blaifa/InternVL3, etc.)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="Number of JSON reports to process (-1 for all).",
    )
    parser.add_argument(
        "--report_dir",
        default=REPORT_DIR,
        help="Directory containing patient JSON reports.",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Base directory to save risk JSON outputs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-patient timeout in seconds (default 600).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    # Ensure model is present
    ensure_ollama_model(args.model)

    # Dynamic subdirectory per model
    model_sanitized = args.model.replace(":", "_").replace("/", "_")
    output_dir = os.path.join(args.output_dir, f"{model_sanitized}", "tot")
    os.makedirs(output_dir, exist_ok=True)

    files = list_json_reports(args.report_dir)
    if args.num != -1:
        files = files[: args.num]

    processed_count = 0
    skipped_timeout = 0
    failed_parse = 0

    for idx, fname in enumerate(files, start=1):
        report_path = os.path.join(args.report_dir, fname)

        # Load patient JSON
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                patient_obj = json.load(f)
        except Exception as e:
            print(f"⚠️  Skipping unreadable JSON {fname}: {e}")
            continue

        patient_id = patient_obj.get("patient_id", os.path.splitext(fname)[0])
        patient_json_str = json.dumps(patient_obj, ensure_ascii=False, indent=2)

        # Call Ollama
        try:
            raw_text = call_ollama(
                model=args.model,
                system_prompt=PROMPT,
                patient_json_str=patient_json_str,
                timeout_seconds=args.timeout,
            )
            if raw_text is None:
                print(f"⏱️ Timeout (> {args.timeout}s) for {fname}. Saving fallback.")
                skipped_timeout += 1
                result = {
                    "patient_id": patient_id,
                    "risk_estimates": {k: 0.0 for k in RISK_KEYS},
                    "time_to_event": {
                        "expected_years": 5.0,
                        "bucket": "> 8 years_or_uncertain",
                    },
                }
            else:
                result, cleaned_block, ok = normalize_result(raw_text, fallback_patient_id=patient_id)
                if not ok:
                    failed_parse += 1

        except Exception as e:
            print(f"❌  Ollama call failed on {fname}: {e}")
            result = {
                "patient_id": patient_id,
                "risk_estimates": {k: 0.0 for k in RISK_KEYS},
                "time_to_event": {
                    "expected_years": 5.0,
                    "bucket": "> 8 years_or_uncertain",
                },
            }
            failed_parse += 1

        # Save result
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"✅ Processed {processed_count} patients")

    print("────────────────────────────────────────────")
    print(f"✅ Total patients processed: {processed_count}")
    print(f"⏭️  Skipped due to timeout: {skipped_timeout}")
    print(f"🧹 JSON parse fallbacks: {failed_parse}")


if __name__ == "__main__":
    main()
