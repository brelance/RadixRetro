import argparse
import dataclasses
import json
import os
import time

import numpy as np
import pandas as pd
import tiktoken

import sglang as sgl
from sglang.srt.server_args import ServerArgs

choices = ["A", "B", "C", "D"]
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def _normalize_subject(name: str) -> str:
    subject = name.strip()
    if subject.endswith("_test.csv"):
        subject = subject[: -len("_test.csv")]
    elif subject.endswith(".csv"):
        subject = subject[: -len(".csv")]
    elif subject.endswith("_test"):
        subject = subject[: -len("_test")]
    return subject


def _pick_subjects(data_dir: str, subjects_arg: str | None) -> list[str]:
    test_dir = os.path.join(data_dir, "test")
    if subjects_arg:
        return [_normalize_subject(s) for s in subjects_arg.split(",") if s.strip()]
    return sorted(
        f.split("_test.csv")[0] for f in os.listdir(test_dir) if f.endswith("_test.csv")
    )


def format_subject(subject: str) -> str:
    return " " + subject.replace("_", " ")


def format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df: pd.DataFrame, subject: str, k: int) -> str:
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def build_few_shot_prompt(
    train_df: pd.DataFrame, subject: str, ntrain: int, token_limit: int
) -> tuple[str, int]:
    k = min(ntrain, train_df.shape[0])
    prompt = gen_prompt(train_df, subject, k)
    while k > 0 and len(tokenizer.encode(prompt)) > token_limit:
        k -= 1
        prompt = gen_prompt(train_df, subject, k)
    return prompt, k


def main(args: argparse.Namespace):
    server_args = ServerArgs.from_cli_args(args)
    engine = sgl.Engine(**dataclasses.asdict(server_args))

    subjects = _pick_subjects(args.data_dir, args.subjects)

    prompts: list[str] = []
    labels: list[str] = []
    num_questions: list[int] = []

    for subject in subjects:
        dev_path = os.path.join(args.data_dir, "dev", f"{subject}_dev.csv")
        test_path = os.path.join(args.data_dir, "test", f"{subject}_test.csv")
        dev_df = pd.read_csv(dev_path, header=None).iloc[: args.ntrain]
        test_df = pd.read_csv(test_path, header=None)
        if args.max_questions > 0:
            test_df = test_df.iloc[: args.max_questions]
        num_questions.append(test_df.shape[0])

        few_shot_examples, shots_used = build_few_shot_prompt(
            dev_df, subject, args.ntrain, args.token_limit
        )
        if shots_used < args.ntrain:
            print(
                f"[warn] subject {subject}: trimmed few-shot examples to {shots_used} to fit token_limit={args.token_limit}"
            )

        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)
            prompts.append(few_shot_examples + prompt_end)
            labels.append(test_df.iloc[i, test_df.shape[1] - 1])

    preds: list[str] = []
    sampling_params = {"temperature": 0, "max_new_tokens": 1}

    tic = time.perf_counter()
    for st in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[st : st + args.batch_size]
        outputs = engine.generate(batch_prompts, sampling_params)
        for out in outputs:
            text = out.get("text", "")
            pred = text.strip()[0] if text.strip() else ""
            preds.append(pred)
    latency = time.perf_counter() - tic

    cors = [pred == label for pred, label in zip(preds, labels)]
    pt = 0
    for subject, num_qs in zip(subjects, num_questions):
        print(
            f"subject: {subject}, #q:{num_qs}, acc: {np.mean(cors[pt : pt + num_qs]):.3f}"
        )
        pt += num_qs

    weighted_acc = np.mean(cors) if cors else 0.0

    if args.raw_result_file:
        with open(args.raw_result_file, "w") as fout:
            for i, (prompt, pred, label) in enumerate(zip(prompts, preds, labels)):
                fout.write(
                    json.dumps(
                        {
                            "prompt_id": i,
                            "prompt": prompt,
                            "output": pred,
                            "label": label,
                            "correct": bool(pred == label),
                        }
                    )
                    + "\n"
                )

    print(f"Total latency: {latency:.3f}")
    print(f"Average accuracy: {weighted_acc:.3f}")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mmlu",
            "backend": "engine",
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(weighted_acc, 3),
            "num_requests": len(prompts),
            "other": {
                "subjects": subjects,
                "batch_size": args.batch_size,
                "ntrain": args.ntrain,
                "token_limit": args.token_limit,
                "max_questions": args.max_questions,
            },
        }
        fout.write(json.dumps(value) + "\n")

    engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="sglang/benchmark/mmlu/data")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument(
        "--subjects",
        type=str,
        help="Comma-separated subject names, e.g. abstract_algebra or abstract_algebra_test.csv. Defaults to all test files.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=-1,
        help="Limit number of questions per subject; -1 uses all.",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=1536,
        help="Few-shot prompt token budget (GPT-3.5 tokenizer).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for offline engine.generate calls.",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="result.jsonl",
        help="Where to append benchmark summaries.",
    )
    parser.add_argument(
        "--raw-result-file",
        type=str,
        help="Optional path to dump per-prompt outputs.",
    )

    # Engine/Server arguments (requires --model-path)
    ServerArgs.add_cli_args(parser)
    cli_args = parser.parse_args()
    main(cli_args)
