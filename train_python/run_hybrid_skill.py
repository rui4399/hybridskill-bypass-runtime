import argparse
import json
from pathlib import Path

from generate_skill_data import JSON_SKILLS, SCHEMA_HINTS
from hybrid_eval_skills import parse_unit_time


DEFAULT_MODEL = "models/eigenskill-smollm2-360m-merged-v2-fp16"
DEFAULT_MAX_NEW_TOKENS = 96


def normalize(text):
    return text.strip().splitlines()[0].strip()


def first_json_value(text):
    text = text.strip()
    start = -1
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start < 0:
        return None

    stack = []
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            expected = "}" if stack[-1] == "{" else "]"
            if char != expected:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def parse_json_output(text):
    candidate = first_json_value(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_prompt(skill, text):
    if skill not in SCHEMA_HINTS:
        raise ValueError(f"unknown skill: {skill}")
    hint = SCHEMA_HINTS[skill]
    json_rule = " For JSON tasks, output a complete JSON object, never a bare span or markdown."
    label_rule = " For label tasks, output exactly the label text."
    return (
        f"<skill:{skill}>\n"
        "You are an EigenSkill micro-kernel for edge inference.\n"
        f"Contract: {hint}\n"
        "Rules: Return exactly one line. No explanation. No extra text."
        f"{json_rule if skill in JSON_SKILLS else label_rule}\n"
        f"Input: {text}\n"
        "Output:"
    )


def deterministic_bypass(skill, text):
    if skill == "unit_time_normalize":
        value = parse_unit_time(text)
        if value is not None:
            return canonical_json(value)
    return None


def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model, torch


def model_predict(skill, text, model_path, max_new_tokens):
    tokenizer, model, torch = load_model(model_path)
    prompt = build_prompt(skill, text) + " "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    if skill in JSON_SKILLS:
        value = parse_json_output(generated)
        if value is not None:
            return canonical_json(value), generated.strip()
        json_text = first_json_value(generated)
        return normalize(json_text or generated), generated.strip()
    return normalize(generated), generated.strip()


def run(skill, text, model_path, max_new_tokens, deterministic_only=False):
    bypass = deterministic_bypass(skill, text)
    if bypass is not None:
        return {
            "skill": skill,
            "input": text,
            "path": "deterministic_bypass",
            "output": bypass,
        }
    if deterministic_only:
        return {
            "skill": skill,
            "input": text,
            "path": "deterministic_only_miss",
            "output": None,
        }
    output, raw = model_predict(skill, text, model_path, max_new_tokens)
    return {
        "skill": skill,
        "input": text,
        "path": "merged_fp16_model",
        "model": str(Path(model_path)),
        "output": output,
        "raw_output": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one EigenSkill v2 hybrid micro-kernel request.")
    parser.add_argument("--skill", required=True, choices=sorted(SCHEMA_HINTS.keys()))
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--deterministic-only", action="store_true")
    args = parser.parse_args()

    result = run(
        skill=args.skill,
        text=args.input,
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        deterministic_only=args.deterministic_only,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
