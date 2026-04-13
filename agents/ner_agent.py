import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import Config


@dataclass
class NERResult:
    """Named entity recognition result for financial documents."""

    entity_text: str
    entity_type: str
    confidence: float
    start: int
    end: int


class NERAgent:
    """Named entity recognition agent using ONNX or fallback HuggingFace models."""

    def __init__(self, config: Config) -> None:
        """Initialize NER agent with ONNX model or fallback pipeline."""
        self.mode = "onnx"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        try:
            from optimum.onnxruntime import ORTModelForTokenClassification
            from transformers import AutoTokenizer

            self.model = ORTModelForTokenClassification.from_pretrained(
                config.ONNX_NER_MODEL_PATH
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.ONNX_NER_MODEL_PATH
            )
            # DistilBERT does not use token_type_ids — tell tokenizer not to generate them
            self.tokenizer.model_input_names = ["input_ids", "attention_mask"]
            self.mode = "onnx"
        except (FileNotFoundError, OSError):
            from transformers import pipeline as hf_pipeline

            # TODO: Replace fallback model with fine-tuned DistilBERT on FiNER-139
            # after running training_notebooks/finetune_ner.ipynb and
            # training_notebooks/export_to_onnx.ipynb on Colab/Kaggle.
            self.pipeline = hf_pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )
            self.mode = "fallback"
            self.tokenizer = None
            self.model = None

    def _map_label(self, raw_label: str) -> str:
        """Map raw model labels to simplified financial entity categories."""
        label = raw_label.upper()
        if "ORG" in label:
            return "ORG"
        if "PER" in label or "PERSON" in label:
            return "PERSON"
        if "MONEY" in label or "MON" in label:
            return "MONEY"
        if "PERCENT" in label:
            return "PERCENTAGE"
        if "DATE" in label or "TIME" in label:
            return "DATE"
        if "TICKER" in label or "STOCK" in label:
            return "TICKER"
        return "METRIC"

    def extract(self, text: str) -> list[NERResult]:
        """Run NER on text and return deduplicated entity spans."""
        if self.mode == "onnx":
            return self._extract_onnx(text)
        return self._extract_fallback(text)

    def _extract_onnx(self, text: str) -> list[NERResult]:
        """Run inference using the ONNX model and decode BIO tags into spans."""
        import numpy as np

        def softmax(logits: np.ndarray) -> np.ndarray:
            """Compute softmax probabilities from raw logits."""
            exp_values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return exp_values / exp_values.sum(axis=-1, keepdims=True)

        tokens = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        # Save offset mapping before removing from inputs —
        # ONNX model does not accept it as an input argument
        offset_mapping = tokens.pop("offset_mapping")

        # DistilBERT does not use token_type_ids —
        # remove before ONNX inference to avoid INVALID_ARGUMENT error
        tokens.pop("token_type_ids", None)

        # ONNX model expects int64 — numpy defaults to int32 on Windows
        import numpy as np
        tokens["input_ids"] = tokens["input_ids"].astype(np.int64)
        tokens["attention_mask"] = tokens["attention_mask"].astype(np.int64)

        outputs = self.model(**tokens)
        logits = outputs.logits[0]
        probabilities = softmax(logits)
        predictions = probabilities.argmax(axis=-1)
        confidences = probabilities.max(axis=-1)
        word_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
        ).word_ids(batch_index=0)

        # Use the saved offset_mapping (not from tokens dict which no longer has it)
        offsets = offset_mapping[0]
        raw_labels = [self.model.config.id2label[pred] for pred in predictions]

        entities: List[NERResult] = []
        current_entity: Optional[NERResult] = None

        for index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            label = raw_labels[index]
            if label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            prefix = label[:2]
            label_type = label[2:]
            start, end = int(offsets[index][0]), int(offsets[index][1])
            confidence = float(confidences[index])
            if (
                prefix == "B"
                or current_entity is None
                or label_type != current_entity.entity_type
            ):
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = NERResult(
                    entity_text=text[start:end],
                    entity_type=label_type,
                    confidence=confidence,
                    start=start,
                    end=end,
                )
            else:
                current_entity = NERResult(
                    entity_text=text[current_entity.start:end],
                    entity_type=current_entity.entity_type,
                    confidence=max(current_entity.confidence, confidence),
                    start=current_entity.start,
                    end=end,
                )

        if current_entity is not None:
            entities.append(current_entity)

        normalized: List[NERResult] = []
        for entity in entities:
            normalized.append(
                NERResult(
                    entity_text=entity.entity_text,
                    entity_type=self._map_label(entity.entity_type),
                    confidence=entity.confidence,
                    start=entity.start,
                    end=entity.end,
                )
            )
        return self._deduplicate_entities(normalized)

    def _extract_fallback(self, text: str) -> list[NERResult]:
        """Run NER using the HuggingFace fallback pipeline."""
        raw_entities = self.pipeline(text)
        entities: List[NERResult] = []
        for item in raw_entities:
            label = item.get("entity_group", item.get("entity", "UNKNOWN"))
            entity_type = self._map_label(label)
            entities.append(
                NERResult(
                    entity_text=item.get("word", "").strip(),
                    entity_type=entity_type,
                    confidence=float(item.get("score", 0.0)),
                    start=int(item.get("start", 0)),
                    end=int(item.get("end", 0)),
                )
            )
        return self._deduplicate_entities(entities)

    def _deduplicate_entities(self, entities: list[NERResult]) -> list[NERResult]:
        """Deduplicate overlapping entity spans keeping the highest confidence span."""
        deduplicated: List[NERResult] = []
        for entity in sorted(
            entities, key=lambda value: (value.start, -value.confidence)
        ):
            skip = False
            for existing in deduplicated:
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    if (
                        entity.entity_text == existing.entity_text
                        and entity.entity_type == existing.entity_type
                    ):
                        skip = True
                        break
            if not skip:
                deduplicated.append(entity)
        return deduplicated

    def extract_from_chunks(self, chunks: list[dict]) -> list[NERResult]:
        """Run NER across all document chunks and return unique highest confidence results."""
        aggregated: List[NERResult] = []
        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else chunk.text
            aggregated.extend(self.extract(text))
        unique: Dict[Tuple[str, str], NERResult] = {}
        for result in aggregated:
            key = (result.entity_text, result.entity_type)
            if key not in unique or unique[key].confidence < result.confidence:
                unique[key] = result
        return list(unique.values())
