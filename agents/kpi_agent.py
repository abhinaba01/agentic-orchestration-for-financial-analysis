import re
from dataclasses import dataclass


@dataclass
class KPIResult:
    """KPI extraction result for a financial document."""

    kpi_name: str
    value: float
    unit: str
    period: str
    source: str
    raw_match: str


class KPIAgent:
    """Extract financial KPIs from text and tables using regex patterns."""

    KPI_PATTERNS = {
        "Revenue": [
            r"(?:total )?(?:net )?(?:revenue|sales|net sales)[^\n\r]*?(\$?[0-9,]+(?:\.[0-9]+)?\s*(?:B|M|T|billion|million|trillion)?(?: dollars?)?)",
        ],
        "Gross Margin": [
            r"gross (?:profit )?margin[^\n\r]*?(?:was\s*)?(?P<value>[0-9]+(?:\.[0-9]+)?\s*(?:%|percent|percentage))",
        ],
        "EPS": [
            r"(?:diluted |basic )?(?:earnings per share|EPS)[^\n\r]*?(\$[0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*per (?:diluted )?share",
        ],
        "Net Income": [
            r"net (?:income|earnings|profit|loss)[^\n\r]*?(\$?[0-9,]+(?:\.[0-9]+)?\s*(?:B|M|T|billion|million|trillion)?(?: dollars?)?)",
        ],
        "Operating Income": [
            r"operating (?:income|profit|earnings)[^\n\r]*?(\$?[0-9,]+(?:\.[0-9]+)?\s*(?:B|M|T|billion|million|trillion)?(?: dollars?)?)",
        ],
    }

    PERIOD_PATTERN = re.compile(
        r"(?:for |in |during )?(Q[1-4] ?\d{4}|fiscal (?:year )?20\d{2}|(?:full[- ]year |annual )?20\d{2})",
        flags=re.IGNORECASE,
    )

    def extract_from_text(self, text: str) -> list[KPIResult]:
        """Extract KPIs from cleaned document text using regex patterns."""
        results: list[KPIResult] = []
        lowered_text = text.lower()
        for kpi_name, patterns in self.KPI_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    raw_match = match.group(0)
                    value_token = match.group(1) if match.lastindex else raw_match
                    parsed = self._parse_value(value_token)
                    if parsed is None:
                        continue
                    period = self._extract_period(raw_match) or self._extract_period(text)
                    results.append(
                        KPIResult(
                            kpi_name=kpi_name,
                            value=parsed[0],
                            unit=parsed[1],
                            period=period or "unknown",
                            source="text",
                            raw_match=raw_match.strip(),
                        )
                    )
        return results

    def extract_from_tables(self, tables: list[dict]) -> list[KPIResult]:
        """Extract KPIs from table rows and headers using header matching."""
        results: list[KPIResult] = []
        for table in tables:
            headers = [header.lower() for header in table.get("headers", [])]
            rows = table.get("rows", [])
            for header_index, header in enumerate(headers):
                for kpi_name in self.KPI_PATTERNS.keys():
                    if any(keyword in header for keyword in kpi_name.lower().split()):
                        for row in rows:
                            if header_index < len(row):
                                raw_value = row[header_index]
                                parsed = self._parse_value(raw_value)
                                if parsed is None:
                                    continue
                                results.append(
                                    KPIResult(
                                        kpi_name=kpi_name,
                                        value=parsed[0],
                                        unit=parsed[1],
                                        period="unknown",
                                        source="table",
                                        raw_match=raw_value.strip(),
                                    )
                                )
        return results

    def extract(self, text: str, tables: list[dict]) -> dict[str, list[KPIResult]]:
        """Extract KPIs from both text and tables and return structured results."""
        text_results = self.extract_from_text(text)
        table_results = self.extract_from_tables(tables)
        combined: dict[str, list[KPIResult]] = {}
        for result in text_results + table_results:
            combined.setdefault(result.kpi_name, []).append(result)
        return combined

    def _parse_value(self, raw_value: str) -> tuple[float, str] | None:
        """Parse a raw numeric value into a float and unit string."""
        cleaned = raw_value.replace(",", "").strip()
        lower_cleaned = cleaned.lower()
        is_per_share = "per share" in lower_cleaned
        if lower_cleaned.endswith(" dollars"):
            cleaned = cleaned[: -len(" dollars")].strip()
            lower_cleaned = cleaned.lower()
        if cleaned.startswith("$"):
            cleaned = cleaned[1:].strip()
            lower_cleaned = cleaned.lower()
        if is_per_share:
            cleaned = lower_cleaned.replace("per share", "").strip()
            lower_cleaned = cleaned
        unit = "USD"
        multiplier = 1.0
        if lower_cleaned.endswith("billion"):
            multiplier = 1_000_000_000.0
            cleaned = cleaned[: -len("billion")].strip()
            unit = "USD_billions"
        elif lower_cleaned.endswith("million"):
            multiplier = 1_000_000.0
            cleaned = cleaned[: -len("million")].strip()
            unit = "USD_millions"
        elif lower_cleaned.endswith("trillion"):
            multiplier = 1_000_000_000_000.0
            cleaned = cleaned[: -len("trillion")].strip()
            unit = "USD_trillions"
        elif cleaned.endswith("B") and cleaned[:-1].replace(".", "", 1).isdigit():
            multiplier = 1_000_000_000.0
            cleaned = cleaned[:-1].strip()
            unit = "USD_billions"
        elif cleaned.endswith("M") and cleaned[:-1].replace(".", "", 1).isdigit():
            multiplier = 1_000_000.0
            cleaned = cleaned[:-1].strip()
            unit = "USD_millions"
        elif cleaned.endswith("T") and cleaned[:-1].replace(".", "", 1).isdigit():
            multiplier = 1_000_000_000_000.0
            cleaned = cleaned[:-1].strip()
            unit = "USD_trillions"
        elif cleaned.endswith("%"):
            try:
                return float(cleaned[:-1].strip()), "percent"
            except ValueError:
                return None
        try:
            value = float(cleaned)
        except ValueError:
            return None
        if is_per_share:
            return value, "per_share"
        return value * multiplier, unit

    def _extract_period(self, text: str) -> str | None:
        """Extract a period description from a text span."""
        match = self.PERIOD_PATTERN.search(text)
        return match.group(1) if match else None
