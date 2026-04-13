import os
import re
from dataclasses import dataclass
from typing import Dict, List

import pdfplumber
from bs4 import BeautifulSoup


@dataclass
class ParsedDocument:
    """Parsed document contents and metadata."""

    doc_id: str
    source_type: str
    raw_text: str
    tables: List[Dict[str, List[str] | str]]
    sections: Dict[str, str]


class DocumentParser:
    """Parser for PDF, HTML, and plain text financial documents."""

    KNOWN_SECTION_KEYWORDS = [
        "management's discussion",
        "risk factors",
        "financial statements",
        "notes to financial",
        "executive summary",
        "results of operations",
        "liquidity and capital resources",
    ]

    SEC_SECTION_MAP = {
        "management's discussion": "MD&A",
        "risk factors": "Risk Factors",
        "financial statements": "Financial Statements",
        "notes to financial": "Notes",
        "executive summary": "Executive Summary",
        "results of operations": "Results of Operations",
        "liquidity and capital resources": "Liquidity and Capital Resources",
    }

    def parse(self, file_path: str) -> ParsedDocument:
        """Detect file type by extension and route to the correct parser."""
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return self._parse_pdf(file_path)
        if extension in {".html", ".htm"}:
            return self._parse_html(file_path)
        if extension == ".txt":
            return self._parse_txt(file_path)
        raise ValueError(f"Unsupported file extension: {extension}")

    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """Parse PDF files, extract full text, tables, and section headings."""
        raw_text_parts: List[str] = []
        tables: List[Dict[str, List[str] | str]] = []
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                raw_text_parts.append(page_text)
                page_tables = page.extract_tables()
                for table_index, raw_table in enumerate(page_tables, start=1):
                    headers: List[str] = []
                    rows: List[List[str]] = []
                    if raw_table:
                        if raw_table[0] is not None:
                            headers = [str(cell).strip() for cell in raw_table[0]]
                            rows = [
                                [str(cell).strip() for cell in row]
                                for row in raw_table[1:]
                            ]
                        else:
                            rows = [
                                [str(cell).strip() for cell in row]
                                for row in raw_table
                            ]
                    caption = f"Page {page_number} Table {table_index}"
                    tables.append({"headers": headers, "rows": rows, "caption": caption})
        raw_text = "\n\n".join(raw_text_parts).strip()
        sections = self._detect_sections(raw_text)
        if not sections:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                if line.isupper() and len(line.split()) <= 8:
                    section_name = line.title()
                    section_lines: List[str] = []
                    for next_line in lines[index + 1 : index + 20]:
                        if next_line.isupper() and len(next_line.split()) <= 8:
                            break
                        section_lines.append(next_line)
                    sections[section_name] = " ".join(section_lines).strip()
        return ParsedDocument(
            doc_id=os.path.basename(file_path),
            source_type="pdf",
            raw_text=raw_text,
            tables=tables,
            sections=sections,
        )

    def _parse_html(self, file_path: str) -> ParsedDocument:
        """Parse HTML files and extract text, tables, and heading sections."""
        with open(file_path, "r", encoding="utf-8") as file_handle:
            html_content = file_handle.read()
        soup = BeautifulSoup(html_content, "lxml")
        text_parts: List[str] = []
        sections: Dict[str, str] = {}
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            if element.name and element.get_text(strip=True):
                text_parts.append(element.get_text(separator=" ", strip=True))
            if element.name and element.name.startswith("h"):
                heading_text = element.get_text(separator=" ", strip=True)
                if heading_text:
                    sections[heading_text] = ""
        tables: List[Dict[str, List[str] | str]] = []
        for table_index, table in enumerate(soup.find_all("table"), start=1):
            headers: List[str] = []
            rows: List[List[str]] = []
            header_row = table.find("tr")
            if header_row:
                headers = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in header_row.find_all(["th", "td"])
                ]
            for row in table.find_all("tr")[1:]:
                row_items = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in row.find_all(["td", "th"])
                ]
                if row_items:
                    rows.append(row_items)
            caption_tag = table.find("caption")
            caption = (
                caption_tag.get_text(strip=True)
                if caption_tag
                else f"Table {table_index}"
            )
            tables.append({"headers": headers, "rows": rows, "caption": caption})
        raw_text = "\n\n".join(text_parts).strip()
        detected_sections = self._detect_sections(raw_text)
        sections.update(detected_sections)
        return ParsedDocument(
            doc_id=os.path.basename(file_path),
            source_type="html",
            raw_text=raw_text,
            tables=tables,
            sections=sections,
        )

    def _parse_txt(self, file_path: str) -> ParsedDocument:
        """Parse plain text files and detect sections from SEC-style patterns."""
        with open(file_path, "r", encoding="utf-8") as file_handle:
            raw_text = file_handle.read()
        sections = self._detect_sections(raw_text)
        return ParsedDocument(
            doc_id=os.path.basename(file_path),
            source_type="txt",
            raw_text=raw_text,
            tables=[],
            sections=sections,
        )

    def _detect_sections(self, text: str) -> Dict[str, str]:
        """Detect known financial sections in text by keyword and return their content."""
        normalized = text.lower()
        sections: Dict[str, str] = {}
        for keyword in self.KNOWN_SECTION_KEYWORDS:
            match = re.search(re.escape(keyword), normalized)
            if not match:
                continue
            section_name = self.SEC_SECTION_MAP.get(keyword, keyword.title())
            start = match.end()
            following_text = normalized[start:]
            next_header = re.search(r"\n\s*[A-Z][A-Za-z\s]+\n", following_text)
            section_body = following_text
            if next_header:
                section_body = following_text[: next_header.start()]
            sections[section_name] = section_body.strip()
        return sections
