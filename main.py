import argparse
import os
import sys

from config import get_config
from preprocessing.parser import DocumentParser
from preprocessing.cleaner import TextCleaner
from preprocessing.chunker import DocumentChunker
from retrieval.embedder import TextEmbedder
from retrieval.vector_store import VectorStore
from agents.ner_agent import NERAgent
from agents.sentiment_agent import SentimentAgent
from agents.kpi_agent import KPIAgent
from agents.rag_agent import RAGAgent
from report.report_generator import ReportGenerator
from tqdm import tqdm


def main() -> None:
    """Run the full financial NLP pipeline end to end from the command line."""
    parser = argparse.ArgumentParser(description="Financial NLP pipeline orchestrator.")
    parser.add_argument("--input", required=True, help="Path to input financial document.")
    parser.add_argument(
        "--query",
        action="append",
        nargs="+",
        help="RAG queries to answer. Can be specified multiple times.",
    )
    parser.add_argument("--output-dir", default="./output", help="Directory to save report files.")
    args = parser.parse_args()

    query_list = ["What are the key financial highlights and risks?"]
    if args.query:
        query_list = [item for group in args.query for item in group]

    try:
        config = get_config()
        parser_stage = DocumentParser()
        cleaner = TextCleaner()
        chunker = DocumentChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        embedder = TextEmbedder(config)
        vector_store = VectorStore(config, embedder)
        ner_agent = NERAgent(config)
        sentiment_agent = SentimentAgent(config)
        kpi_agent = KPIAgent()
        rag_agent = RAGAgent(config, vector_store)
        report_generator = ReportGenerator()

        with tqdm(total=1, desc="Stage 1 — Parse", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            parsed_doc = parser_stage.parse(args.input)
            pbar.update(1)

        with tqdm(total=1, desc="Stage 2 — Clean", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            parsed_doc.raw_text = cleaner.clean(parsed_doc.raw_text)
            parsed_doc.sections = {
                name: cleaner.clean(section_text)
                for name, section_text in parsed_doc.sections.items()
            }
            pbar.update(1)

        with tqdm(total=1, desc="Stage 3 — Chunk", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            chunks = chunker.chunk_document(parsed_doc.doc_id, parsed_doc.sections, parsed_doc.raw_text)
            section_count = len(parsed_doc.sections) if parsed_doc.sections else 1
            print(f"Created {len(chunks)} chunks across {section_count} sections")
            pbar.update(1)

        with tqdm(total=1, desc="Stage 4 — Index", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            vector_store.index_document(parsed_doc.doc_id, chunks)
            pbar.update(1)

        with tqdm(total=1, desc="Stage 5 — NER", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            ner_results = ner_agent.extract_from_chunks([chunk.__dict__ for chunk in chunks])
            print(f"Found {len(ner_results)} unique entities")
            pbar.update(1)

        with tqdm(total=1, desc="Stage 6 — Sentiment", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            sentiment_results = sentiment_agent.analyze_document([chunk.__dict__ for chunk in chunks])
            pbar.update(1)

        with tqdm(total=1, desc="Stage 7 — KPI Extraction", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            kpi_results = kpi_agent.extract(parsed_doc.raw_text, parsed_doc.tables)
            print(f"Extracted KPIs: {', '.join(sorted(kpi_results.keys())) if kpi_results else 'none'}")
            pbar.update(1)

        with tqdm(total=1, desc="Stage 8 — RAG", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            rag_answers = rag_agent.batch_answer(query_list, doc_ids=[parsed_doc.doc_id])
            pbar.update(1)

        with tqdm(total=1, desc="Stage 9 — Report", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            os.makedirs(args.output_dir, exist_ok=True)
            report = report_generator.generate_report(
                parsed_doc.doc_id,
                parsed_doc,
                ner_results,
                sentiment_results,
                kpi_results,
                rag_answers,
            )
            json_path = os.path.join(args.output_dir, f"{parsed_doc.doc_id}_report.json")
            markdown_path = os.path.join(args.output_dir, f"{parsed_doc.doc_id}_report.md")
            report_generator.to_json(report, json_path)
            report_generator.to_markdown(report, markdown_path)
            print(f"Pipeline complete. Reports saved to {args.output_dir}")
            pbar.update(1)

    except Exception as error:
        print(f"An error occurred during pipeline execution: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
