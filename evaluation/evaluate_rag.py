import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

from config import get_config
from openai import OpenAI
from retrieval.embedder import TextEmbedder
from retrieval.vector_store import VectorStore
from agents.rag_agent import RAGAgent


def load_qa_pairs(file_path: str) -> List[Dict[str, str]]:
    """Load and validate FinanceBench-style QA pairs from a JSON file."""
    if not os.path.exists(file_path):
        raise ValueError(f"QA file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise ValueError("QA file must contain a list of question objects.")
    validated_pairs: List[Dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"QA pair at index {index} must be a JSON object.")
        for field in ["question", "answer", "doc_id"]:
            if field not in item:
                raise ValueError(
                    f"QA pair at index {index} is missing required field '{field}'."
                )
            if not isinstance(item[field], str):
                raise ValueError(
                    f"QA pair at index {index} field '{field}' must be a string."
                )
        validated_pairs.append(
            {
                "question": item["question"].strip(),
                "answer": item["answer"].strip(),
                "doc_id": item["doc_id"].strip(),
            }
        )
    return validated_pairs


def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score between prediction and ground truth."""
    return 1.0 if _normalize_text(prediction) == _normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level precision, recall, and F1 score for a prediction."""
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    truth_counts = Counter(truth_tokens)
    common_tokens = sum(
        min(pred_counts[token], truth_counts[token]) for token in pred_counts
    )
    precision = common_tokens / len(pred_tokens)
    recall = common_tokens / len(truth_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_gpt_judge_score(
    question: str,
    prediction: str,
    ground_truth: str,
    client: OpenAI,
) -> int:
    """Call GPT-4o to score answer quality on a 1-5 scale."""
    system_prompt = (
        "You are an expert financial analyst evaluating answer quality. "
        "Rate the predicted answer from 1 to 5 where: "
        "5 = Correct and complete, matches ground truth in all key facts. "
        "4 = Mostly correct, minor omissions or slight differences. "
        "3 = Partially correct, some key facts right but important gaps. "
        "2 = Mostly incorrect but shows some understanding. "
        "1 = Completely wrong or irrelevant. "
        "Respond with ONLY the integer score, nothing else."
    )
    user_message = (
        f"Question: {question}\nGround Truth: {ground_truth}\nPredicted Answer: {prediction}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content.strip()
        score = int(re.search(r"\b([1-5])\b", content).group(1))
        return score
    except Exception:
        return 0


def run_evaluation(
    qa_pairs: List[Dict[str, str]],
    rag_agent: RAGAgent,
    openai_client: OpenAI,
    use_gpt_judge: bool = True,
) -> Dict[str, Any]:
    """Evaluate each QA pair and aggregate EM, token F1, and GPT judge scores."""
    per_question: List[Dict[str, Any]] = []
    total_exact_match = 0.0
    total_token_f1 = 0.0
    total_gpt_judge = 0.0
    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]
        prediction = rag_agent.answer(question).answer
        exact_match = compute_exact_match(prediction, ground_truth)
        token_f1 = compute_token_f1(prediction, ground_truth)
        gpt_score = 0
        if use_gpt_judge:
            gpt_score = compute_gpt_judge_score(
                question, prediction, ground_truth, openai_client
            )
        per_question.append(
            {
                "question": question,
                "doc_id": qa_pair["doc_id"],
                "prediction": prediction,
                "ground_truth": ground_truth,
                "exact_match": exact_match,
                "token_f1": token_f1,
                "gpt_judge_score": gpt_score,
            }
        )
        total_exact_match += exact_match
        total_token_f1 += token_f1
        total_gpt_judge += gpt_score
    num_questions = len(per_question)
    return {
        "num_questions": num_questions,
        "mean_exact_match": total_exact_match / num_questions if num_questions else 0.0,
        "mean_token_f1": total_token_f1 / num_questions if num_questions else 0.0,
        "mean_gpt_judge_score": total_gpt_judge / num_questions if num_questions else 0.0,
        "per_question": per_question,
    }


def print_evaluation_table(results: Dict[str, Any]) -> None:
    """Print a formatted results table with question metrics and average scores."""
    header = f"{'Question':60} | {'EM':>4} | {'Token F1':>7} | {'GPT Score':>9}"
    divider = "-" * len(header)
    print(header)
    print(divider)
    for item in results["per_question"]:
        question = item["question"]
        truncated_question = question[:57] + "..." if len(question) > 60 else question
        print(
            f"{truncated_question:60} | {item['exact_match']:4.2f} | {item['token_f1']:7.2f} | {item['gpt_judge_score']:9d}"
        )
    print(divider)
    print(
        f"{'AVERAGE':60} | {results['mean_exact_match']:4.2f} | {results['mean_token_f1']:7.2f} | {results['mean_gpt_judge_score']:9.2f}"
    )


def _save_results(results: Dict[str, Any], qa_file_path: str) -> str:
    """Save evaluation results to a JSON file adjacent to the input QA file."""
    base_name = os.path.splitext(os.path.basename(qa_file_path))[0]
    output_file_name = f"{base_name}_evaluation_results.json"
    output_path = os.path.join(os.path.dirname(qa_file_path), output_file_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    return output_path


def main() -> None:
    """Parse command line arguments and run the RAG QA evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline on FinanceBench-style QA pairs."
    )
    parser.add_argument(
        "--qa-file",
        required=True,
        help="Path to the JSON file containing QA pairs.",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional document id filter to evaluate only matching QA pairs.",
    )
    parser.add_argument(
        "--no-gpt-judge",
        action="store_true",
        help="Disable GPT judge scoring.",
    )
    args = parser.parse_args()

    config = get_config()
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    embedder = TextEmbedder(config)
    vector_store = VectorStore(config, embedder)
    rag_agent = RAGAgent(config, vector_store)

    qa_pairs = load_qa_pairs(args.qa_file)
    if args.doc_id:
        qa_pairs = [pair for pair in qa_pairs if pair["doc_id"] == args.doc_id]
    if not qa_pairs:
        raise ValueError("No QA pairs found after applying the requested filters.")

    results = run_evaluation(
        qa_pairs=qa_pairs,
        rag_agent=rag_agent,
        openai_client=openai_client,
        use_gpt_judge=not args.no_gpt_judge,
    )
    print_evaluation_table(results)
    saved_path = _save_results(results, args.qa_file)
    print(f"Saved evaluation results to {saved_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
