import os
from dataclasses import dataclass
from typing import Dict, List, Union

from config import Config


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a text chunk."""

    label: str
    confidence: float
    scores: Dict[str, float]
    chunk_id: str


class SentimentAgent:
    """Sentiment analysis agent using ONNX or fallback HuggingFace models."""

    def __init__(self, config: Config) -> None:
        """Initialize the sentiment agent by loading ONNX or fallback models."""
        self.mode = "onnx"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            self.model = ORTModelForSequenceClassification.from_pretrained(
                config.ONNX_SENTIMENT_MODEL_PATH
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.ONNX_SENTIMENT_MODEL_PATH
            )
            # Do NOT strip token_type_ids — FinBERT is BERT-based and requires them
            self.mode = "onnx"
        except (FileNotFoundError, OSError):
            from transformers import AutoTokenizer, pipeline

            # TODO: Replace with fine-tuned FinBERT after running Colab notebooks
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer=self.tokenizer,
                return_all_scores=True,
            )
            self.mode = "fallback"
            self.model = None

    def analyze_chunk(self, text: str, chunk_id: str = "") -> SentimentResult:
        """Analyze the sentiment of a single text chunk."""
        if self.mode == "onnx":
            return self._analyze_chunk_onnx(text, chunk_id)
        return self._analyze_chunk_fallback(text, chunk_id)

    def _truncate_to_tokens(self, text: str) -> str:
        """Truncate the input text to the first 512 tokens according to the tokenizer."""
        if self.tokenizer is None:
            return text
        token_ids = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=512,
        )
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _analyze_chunk_onnx(self, text: str, chunk_id: str) -> SentimentResult:
        """Analyze sentiment for a chunk using the ONNX sequence classification model."""
        import numpy as np

        truncated_text = self._truncate_to_tokens(text)
        tokens = self.tokenizer(
            truncated_text,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        # ONNX model expects int64 — numpy defaults to int32 on Windows
        # FinBERT requires token_type_ids so we keep it and cast all to int64
        tokens["input_ids"] = tokens["input_ids"].astype(np.int64)
        tokens["attention_mask"] = tokens["attention_mask"].astype(np.int64)
        if "token_type_ids" in tokens:
            tokens["token_type_ids"] = tokens["token_type_ids"].astype(np.int64)

        outputs = self.model(**tokens)
        logits = outputs.logits[0]
        probabilities = self._softmax(logits)
        label_index = int(np.argmax(probabilities))
        raw_label = self.model.config.id2label[label_index]
        normalized_label = self._normalize_label(raw_label)
        scores = {
            self._normalize_label(self.model.config.id2label[index]): float(probabilities[index])
            for index in range(len(probabilities))
        }
        normalized_scores = self._normalize_scores(scores)
        confidence = float(normalized_scores[normalized_label])
        return SentimentResult(
            label=normalized_label,
            confidence=confidence,
            scores=normalized_scores,
            chunk_id=chunk_id,
        )

    def _analyze_chunk_fallback(self, text: str, chunk_id: str) -> SentimentResult:
        """Analyze sentiment for a chunk using the fallback HuggingFace pipeline."""
        truncated_text = self._truncate_to_tokens(text)
        raw_results = self.pipeline(truncated_text)
        scores = {
            self._normalize_label(item["label"]): float(item["score"])
            for item in raw_results
        }
        normalized_scores = self._normalize_scores(scores)
        best_label = max(normalized_scores, key=normalized_scores.get)
        confidence = float(normalized_scores[best_label])
        return SentimentResult(
            label=best_label,
            confidence=confidence,
            scores=normalized_scores,
            chunk_id=chunk_id,
        )

    def _softmax(self, logits: Union[List[float], "object"]) -> List[float]:
        """Compute softmax probabilities from raw output logits."""
        import numpy as np

        array_logits = np.array(logits, dtype=float)
        exp_logits = np.exp(array_logits - np.max(array_logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities.tolist()

    def _normalize_label(self, label: str) -> str:
        """Normalize a raw label to a supported sentiment label."""
        label_lower = label.lower()
        if "positive" in label_lower or label_lower == "label_2":
            return "positive"
        if "negative" in label_lower or label_lower == "label_0":
            return "negative"
        return "neutral"

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Ensure the scores dictionary contains all supported sentiment labels."""
        normalized = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for label, value in scores.items():
            normalized[self._normalize_label(label)] = float(value)
        return normalized

    def analyze_document(self, chunks: List[Dict]) -> Dict[str, object]:
        """Analyze sentiment for a list of document chunks and return aggregated metrics."""
        chunk_results: List[SentimentResult] = []
        weighted_score_totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        weighted_confidence_total = 0.0
        total_weight = 0.0
        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else chunk.text
            chunk_id = chunk["chunk_id"] if isinstance(chunk, dict) else chunk.chunk_id
            token_count = float(
                chunk.get("token_count", 0)
                if isinstance(chunk, dict)
                else getattr(chunk, "token_count", 0)
            )
            if token_count <= 0.0:
                token_count = 1.0
            chunk_result = self.analyze_chunk(text, chunk_id)
            chunk_results.append(chunk_result)
            for label, score in chunk_result.scores.items():
                weighted_score_totals[label] += score * token_count
            weighted_confidence_total += chunk_result.confidence * token_count
            total_weight += token_count
        if total_weight <= 0.0:
            total_weight = 1.0
        sentiment_breakdown = {
            label: float(weighted_score_totals[label] / total_weight)
            for label in weighted_score_totals
        }
        overall_sentiment = max(sentiment_breakdown, key=sentiment_breakdown.get)
        overall_confidence = float(weighted_confidence_total / total_weight)
        return {
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "chunk_results": chunk_results,
            "sentiment_breakdown": sentiment_breakdown,
        }
