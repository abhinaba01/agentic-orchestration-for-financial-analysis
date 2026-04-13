import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from preprocessing.parser import ParsedDocument
from agents.ner_agent import NERResult
from agents.rag_agent import RAGResult


@dataclass
class FinancialReport:
    """Dataclass representing a generated financial analysis report."""

    doc_id: str
    company_name: str
    report_date: str
    overall_sentiment: str
    sentiment_breakdown: dict
    key_entities: dict
    kpis: dict
    rag_insights: list
    summary: str
    generated_at: str


class ReportGenerator:
    """Generate JSON and Markdown reports from agent outputs."""

    def generate_report(
        self,
        doc_id: str,
        parsed_doc: ParsedDocument,
        ner_results: List[NERResult],
        sentiment_results: dict,
        kpi_results: dict,
        rag_answers: List[RAGResult],
    ) -> FinancialReport:
        """Build a FinancialReport dataclass from parsing and agent outputs."""
        org_entities = [
            result.entity_text
            for result in ner_results
            if result.entity_type == "ORG"
        ]
        company_name = (
            max(set(org_entities), key=org_entities.count)
            if org_entities
            else "unknown"
        )
        date_entities = [
            result.entity_text
            for result in ner_results
            if result.entity_type == "DATE"
        ]
        report_date = date_entities[0] if date_entities else "unknown"

        key_entities: Dict[str, List[str]] = {
            "ORG": [],
            "PERSON": [],
            "TICKER": [],
            "MONEY": [],
            "METRIC": [],
        }
        for result in ner_results:
            if result.entity_type in key_entities:
                if result.entity_text not in key_entities[result.entity_type]:
                    key_entities[result.entity_type].append(result.entity_text)

        top_kpis = []
        for kpi_name, values in kpi_results.items():
            if values:
                top_kpis.append((kpi_name, values[0]))

        summary_kpis = ", ".join(
            [f"{name}={result.raw_match}" for name, result in top_kpis[:3]]
        )
        sections_analyzed = (
            ", ".join(parsed_doc.sections.keys())
            if parsed_doc.sections
            else "full document"
        )
        summary = (
            f"The financial analysis for {company_name} shows an overall sentiment of "
            f"{sentiment_results.get('overall_sentiment', 'unknown')}. "
            f"Key metrics discovered include {summary_kpis or 'no KPI matches found'}. "
            f"The RAG pipeline generated {len(rag_answers)} insights. "
            f"Sections analyzed include {sections_analyzed}."
        )

        # Safely serialize kpi_results — values may be dataclasses or dicts
        serialized_kpis: Dict[str, List[dict]] = {}
        for kpi_name, values in kpi_results.items():
            serialized_values = []
            for item in values:
                if dataclasses.is_dataclass(item):
                    serialized_values.append(dataclasses.asdict(item))
                elif isinstance(item, dict):
                    serialized_values.append(item)
                else:
                    serialized_values.append({"value": str(item)})
            serialized_kpis[kpi_name] = serialized_values

        # Safely serialize rag_answers
        serialized_rag: List[dict] = []
        for answer in rag_answers:
            if dataclasses.is_dataclass(answer):
                serialized_rag.append(dataclasses.asdict(answer))
            elif isinstance(answer, dict):
                serialized_rag.append(answer)
            else:
                serialized_rag.append({"answer": str(answer)})

        return FinancialReport(
            doc_id=doc_id,
            company_name=company_name,
            report_date=report_date,
            overall_sentiment=sentiment_results.get("overall_sentiment", "unknown"),
            sentiment_breakdown=sentiment_results.get("sentiment_breakdown", {}),
            key_entities=key_entities,
            kpis=serialized_kpis,
            rag_insights=serialized_rag,
            summary=summary,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )

    def to_json(self, report: FinancialReport, output_path: str) -> None:
        """Serialize the financial report to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(dataclasses.asdict(report), file_handle, indent=2)
        print(f"Report saved to {output_path}")

    def to_markdown(self, report: FinancialReport, output_path: str) -> None:
        """Write the financial report to a Markdown file."""
        lines: List[str] = []
        lines.append(f"# Financial Analysis Report — {report.company_name}")
        lines.append(
            f"Generated: {report.generated_at}  |  Document ID: {report.doc_id}"
        )
        lines.append("")
        lines.append("## Executive Summary")
        lines.append(report.summary)
        lines.append("")
        lines.append("## Key Performance Indicators")
        lines.append("KPI | Value | Unit | Period | Source")
        lines.append("--- | --- | --- | --- | ---")
        for kpi_name, results in report.kpis.items():
            for result in results:
                lines.append(
                    f"{kpi_name} | {result.get('value')} | "
                    f"{result.get('unit')} | {result.get('period')} | "
                    f"{result.get('source')}"
                )
        lines.append("")
        lines.append("## Sentiment Analysis")
        lines.append(f"Overall Sentiment: **{report.overall_sentiment}**")
        lines.append("")
        lines.append("Sentiment | Score")
        lines.append("--- | ---")
        for label in ["positive", "negative", "neutral"]:
            score = report.sentiment_breakdown.get(label, 0.0)
            lines.append(f"{label.capitalize()} | {score:.2f}")
        lines.append("")
        lines.append("## Key Entities Identified")
        lines.append("### Organizations")
        lines.append(", ".join(report.key_entities.get("ORG", [])) or "None")
        lines.append("")
        lines.append("### People")
        lines.append(", ".join(report.key_entities.get("PERSON", [])) or "None")
        lines.append("")
        lines.append("### Tickers")
        lines.append(", ".join(report.key_entities.get("TICKER", [])) or "None")
        lines.append("")
        lines.append("### Financial Figures")
        lines.append(
            ", ".join(
                report.key_entities.get("MONEY", [])
                + report.key_entities.get("METRIC", [])
            )
            or "None"
        )
        lines.append("")
        lines.append("## RAG Insights")
        for answer in report.rag_insights:
            lines.append(f"### Query: {answer.get('query', '')}")
            lines.append(f"**Answer:** {answer.get('answer', '')}")
            lines.append(
                f"**Chain of Thought:** {answer.get('chain_of_thought', '')}"
            )
            evidence_lines = []
            for index, evidence in enumerate(
                answer.get("evidence_chunks_used", []), start=1
            ):
                evidence_lines.append(
                    f"[Evidence {index}] Doc: {evidence.get('doc_id', '')} "
                    f"/ Section: {evidence.get('section', '')}"
                )
            lines.append("**Evidence used:**")
            lines.extend(evidence_lines or ["No evidence available."])
            lines.append("")
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(lines))
        print(f"Markdown report saved to {output_path}")
