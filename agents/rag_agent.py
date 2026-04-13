from dataclasses import dataclass
from typing import Dict, List, Optional

from config import Config
from openai import OpenAI


@dataclass
class RAGResult:
    """Result of a retrieval-augmented generation query."""

    query: str
    answer: str
    chain_of_thought: str
    evidence_chunks_used: List[Dict]
    model: str
    total_tokens_used: int


class RAGAgent:
    """Retrieval-augmented generation agent using OpenAI chat completions."""

    def __init__(self, config: Config, vector_store: object) -> None:
        """Initialize the RAG agent with an OpenAI client and a vector store."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_store = vector_store
        self.model = config.GPT_MODEL

    def answer(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> RAGResult:
        """Retrieve evidence chunks and answer a query using GPT-4o."""
        evidence_chunks = self.vector_store.query(
            query, top_k=top_k, doc_ids=doc_ids
        )
        evidence_lines: List[str] = []
        for index, chunk in enumerate(evidence_chunks, start=1):
            section = chunk.get("section", "")
            doc_id = chunk.get("doc_id", "")
            chunk_text = chunk.get("chunk_text", "")
            evidence_lines.append(
                f"[Evidence {index}] (Section: {section}, Doc: {doc_id})\n{chunk_text}"
            )
        user_message = "\n\n".join(evidence_lines) + f"\n\nQuestion: {query}"
        system_message = (
            "You are a precise financial analyst assistant. "
            "You will be given evidence chunks from financial documents and a question. "
            "Think step by step. Show your calculation explicitly if the question requires math. "
            "Cite which Evidence number supports each step of your reasoning using [Evidence N] notation. "
            "If the evidence does not contain enough information to answer confidently, say so explicitly. "
            "Do not hallucinate numbers not present in the evidence."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        full_text = response.choices[0].message.content.strip()
        split_paragraphs = [
            paragraph.strip()
            for paragraph in full_text.split("\n\n")
            if paragraph.strip()
        ]
        answer_text = split_paragraphs[-1] if split_paragraphs else full_text
        total_tokens_used = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens_used = int(getattr(usage, "total_tokens", 0))
        return RAGResult(
            query=query,
            answer=answer_text,
            chain_of_thought=full_text,
            evidence_chunks_used=evidence_chunks,
            model=self.model,
            total_tokens_used=total_tokens_used,
        )

    def batch_answer(
        self,
        queries: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> List[RAGResult]:
        """Answer a batch of queries and return a list of RAG results."""
        results: List[RAGResult] = []
        total = len(queries)
        for index, query in enumerate(queries, start=1):
            print(f"Answering query {index}/{total}: {query[:60]}...")
            result = self.answer(query, doc_ids=doc_ids)
            results.append(result)
        return results
