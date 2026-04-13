from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass
class Chunk:
    """Chunk object representing a segment of document text."""

    chunk_id: str
    doc_id: str
    section: str
    text: str
    token_count: int


class DocumentChunker:
    """Chunk financial documents into token-based windows."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens for given text using cl100k_base encoding."""
        return len(self.encoding.encode(text))

    def chunk_document(
        self,
        doc_id: str,
        sections: dict[str, str],
        raw_text: str,
    ) -> list[Chunk]:
        """Chunk a document into overlapping token windows organized by section."""
        if sections:
            section_items = list(sections.items())
        else:
            section_items = [("full_document", raw_text)]
        chunks: list[Chunk] = []
        for section_name, section_text in section_items:
            tokens = self.encoding.encode(section_text)
            step = max(self.chunk_size - self.chunk_overlap, 1)
            start = 0
            index = 0
            while start < len(tokens):
                window = tokens[start : start + self.chunk_size]
                if not window:
                    break
                chunk_text = self.encoding.decode(window)
                token_count = len(window)
                if token_count >= 20:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}__{section_name}__{index:04d}",
                            doc_id=doc_id,
                            section=section_name,
                            text=chunk_text,
                            token_count=token_count,
                        )
                    )
                    index += 1
                start += step
        return chunks
