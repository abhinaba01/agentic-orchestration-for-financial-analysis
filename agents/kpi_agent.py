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

    # ── Amount suffix pattern (reused in all patterns) ───────────────────────
    _AMT = (
        r"\$?\s*"
        r"(\d[\d,]*(?:\.\d+)?)"
        r"\s*"
        r"(billion|million|trillion|B|M|T)?"
        r"(?:\s*(?:dollars?|USD))?"
    )

    KPI_PATTERNS = {
        # ── Income statement ─────────────────────────────────────────────────
        "Revenue": [
            r"(?:total\s+)?(?:net\s+)?(?:revenue|sales|net\s+sales)"
            r"[^\n\r]{0,40}?" + _AMT,
        ],
        "Gross Margin": [
            r"gross\s+(?:profit\s+)?margin"
            r"[^\n\r]{0,40}?"
            r"([0-9]+(?:\.[0-9]+)?)\s*(%|percent|percentage\s+points?|pp)",
            r"gross\s+margin\s+(?:was|of)\s+"
            r"([0-9]+(?:\.[0-9]+)?)\s*(%|percent)",
        ],
        "EPS": [
            r"(?:diluted\s+|basic\s+)?(?:earnings\s+per\s+share|EPS)"
            r"[^\n\r]{0,40}?\$([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s+per\s+(?:diluted\s+)?share",
            r"\$([0-9]+(?:\.[0-9]+)?)\s+per\s+(?:diluted\s+)?share",
        ],
        "Net Income": [
            r"net\s+(?:income|earnings|profit)"
            r"[^\n\r]{0,40}?" + _AMT,
        ],
        "Net Loss": [
            r"net\s+(?:loss|quarterly\s+loss|second[- ]quarter\s+loss|annual\s+loss)"
            r"[^\n\r]{0,40}?" + _AMT,
            r"(?:reported|recorded|posted)\s+(?:a\s+)?(?:net\s+)?loss\s+of\s+" + _AMT,
            r"loss\s+of\s+" + _AMT,
        ],
        "Operating Income": [
            r"operating\s+(?:income|profit|earnings)"
            r"[^\n\r]{0,40}?" + _AMT,
        ],
        # ── Balance sheet ────────────────────────────────────────────────────
        "Total Assets": [
            r"(?:total\s+)?assets?"
            r"[^\n\r]{0,30}?" + _AMT,
            r"(?:firm|company|bank)\s+(?:had|with|held|reported)\s+"
            r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)?\s*in\s+assets",
        ],
        "Total Debt": [
            r"(?:total\s+)?(?:debt|liabilities?|obligations?)"
            r"[^\n\r]{0,30}?" + _AMT,
            r"(?:firm|company)\s+(?:had|with|held)\s+"
            r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)?\s*in\s+"
            r"(?:debt|liabilities)",
        ],
        "Leverage Ratio": [
            r"leverage\s+ratio[^\n\r]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*(?::\s*1|x|times)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*:\s*1\s+leverage",
            r"leverage[^\n\r]{0,20}?([0-9]+(?:\.[0-9]+)?)\s*to\s*1",
        ],
        # ── Cash & liquidity ─────────────────────────────────────────────────
        "Cash & Equivalents": [
            r"cash\s+(?:and\s+cash\s+equivalents?|equivalents?)"
            r"[^\n\r]{0,30}?" + _AMT,
            r"balances?\s+of\s+cash[^\n\r]{0,30}?" + _AMT,
        ],
        "Creditor Recovery": [
            r"return(?:ing|ed)?\s+" + _AMT + r"\s+to\s+creditors",
            r"(?:returned|paid|distributed)\s+approximately\s+" + _AMT
            + r"\s+to\s+creditors",
        ],
        # ── Off-balance / restructuring ───────────────────────────────────────
        "Off-Balance Sheet": [
            r"(?:move?d?|shift(?:ed)?|remov(?:ed?)?|tak(?:en|ing))\s+"
            r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)?\s+"
            r"(?:in\s+)?(?:debt|assets?|liabilities?)\s+off",
            r"off\s+(?:its\s+)?balance\s+sheet[^\n\r]{0,60}?" + _AMT,
            r"repo\s+105[^\n\r]{0,80}?" + _AMT,
        ],
        # ── Stock / market ────────────────────────────────────────────────────
        "Stock Price Decline": [
            r"stock\s+(?:price\s+)?(?:plunged?|fell?|dropped?|declined?)\s+"
            r"(?:by\s+)?(?:over\s+|more\s+than\s+)?"
            r"([0-9]+(?:\.[0-9]+)?)\s*(%|percent)",
            r"(?:plunged?|fell?|dropped?)\s+(?:by\s+)?(?:over\s+)?"
            r"([0-9]+(?:\.[0-9]+)?)\s*(%|percent)",
        ],
        "Market Drop": [
            r"dow\s+jones[^\n\r]{0,40}?fell?\s+"
            r"([0-9,]+(?:\.[0-9]+)?)\s+points?",
            r"(?:fell?|dropped?|declined?)\s+([0-9,]+(?:\.[0-9]+)?)\s+points?",
        ],
    }

    PERIOD_PATTERN = re.compile(
        r"(?:for |in |during )?"
        r"(Q[1-4]\s?\d{4}"
        r"|fiscal\s+(?:year\s+)?20\d{2}"
        r"|(?:full[- ]year\s+|annual\s+)?20\d{2}"
        r"|(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?20\d{2}"
        r"|FY\s*20\d{2})",
        flags=re.IGNORECASE,
    )

    def extract_from_text(self, text: str) -> list[KPIResult]:
        """Extract KPIs from cleaned document text using regex patterns."""
        results: list[KPIResult] = []
        for kpi_name, patterns in self.KPI_PATTERNS.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                        raw_match = match.group(0)

                        # Leverage ratio and percentage KPIs handled separately
                        if kpi_name in ("Leverage Ratio", "Stock Price Decline", "Market Drop"):
                            parsed = self._parse_special(kpi_name, match)
                        else:
                            parsed = self._parse_from_match(match)

                        if parsed is None:
                            continue

                        period = (
                            self._extract_period(raw_match)
                            or self._extract_period(text[:500])
                            or "unknown"
                        )
                        results.append(
                            KPIResult(
                                kpi_name=kpi_name,
                                value=parsed[0],
                                unit=parsed[1],
                                period=period,
                                source="text",
                                raw_match=raw_match.strip(),
                            )
                        )
                except re.error:
                    continue
        return results

    def extract_from_tables(self, tables: list[dict]) -> list[KPIResult]:
        """Extract KPIs from table rows and headers using header matching."""
        results: list[KPIResult] = []
        for table in tables:
            headers = [h.lower() for h in table.get("headers", [])]
            rows = table.get("rows", [])
            for header_index, header in enumerate(headers):
                for kpi_name in self.KPI_PATTERNS.keys():
                    keywords = kpi_name.lower().split()
                    if any(kw in header for kw in keywords):
                        for row in rows:
                            if header_index < len(row):
                                raw_value = row[header_index]
                                parsed = self._parse_value(str(raw_value))
                                if parsed is None:
                                    continue
                                results.append(
                                    KPIResult(
                                        kpi_name=kpi_name,
                                        value=parsed[0],
                                        unit=parsed[1],
                                        period="unknown",
                                        source="table",
                                        raw_match=str(raw_value).strip(),
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_from_match(self, match: re.Match) -> tuple[float, str] | None:
        """Extract value and unit from a regex match object."""
        groups = [g for g in match.groups() if g is not None]
        if not groups:
            return self._parse_value(match.group(0))
        # Try each group from left looking for a parseable number
        for g in groups:
            result = self._parse_value(g)
            if result is not None:
                return result
        return None

    def _parse_special(
        self, kpi_name: str, match: re.Match
    ) -> tuple[float, str] | None:
        """Handle KPIs that are percentages, ratios or point values."""
        groups = [g for g in match.groups() if g is not None]
        if not groups:
            return None
        try:
            value = float(groups[0].replace(",", ""))
        except ValueError:
            return None
        if kpi_name == "Leverage Ratio":
            return value, "ratio_to_1"
        if kpi_name == "Stock Price Decline":
            return value, "percent"
        if kpi_name == "Market Drop":
            return value, "points"
        return value, "USD"

    def _parse_value(self, raw_value: str) -> tuple[float, str] | None:
        """Parse a raw numeric string into (float, unit) handling B/M/T suffixes."""
        cleaned = raw_value.replace(",", "").strip()
        lower = cleaned.lower()

        # Strip trailing qualifiers
        for suffix in (" dollars", " usd"):
            if lower.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                lower = cleaned.lower()

        # Strip leading currency symbol
        if cleaned.startswith("$"):
            cleaned = cleaned[1:].strip()
            lower = cleaned.lower()

        is_per_share = "per share" in lower
        if is_per_share:
            cleaned = re.sub(r"per\s+(?:diluted\s+)?share", "", cleaned,
                             flags=re.IGNORECASE).strip()
            lower = cleaned.lower()

        # Percentage
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1].strip()), "percent"
            except ValueError:
                return None

        unit = "USD"
        multiplier = 1.0

        suffixes = [
            ("trillion", 1e12,  "USD_trillions"),
            ("billion",  1e9,   "USD_billions"),
            ("million",  1e6,   "USD_millions"),
        ]
        for word, mult, u in suffixes:
            if lower.endswith(word):
                multiplier, unit = mult, u
                cleaned = cleaned[: -len(word)].strip()
                lower = cleaned.lower()
                break
        else:
            for letter, mult, u in [("T", 1e12, "USD_trillions"),
                                     ("B", 1e9,  "USD_billions"),
                                     ("M", 1e6,  "USD_millions")]:
                if cleaned.endswith(letter) and cleaned[:-1].replace(".", "", 1).isdigit():
                    multiplier, unit = mult, u
                    cleaned = cleaned[:-1].strip()
                    break

        try:
            value = float(cleaned)
        except ValueError:
            return None

        if is_per_share:
            return value, "per_share"
        return value * multiplier, unit

    def _extract_period(self, text: str) -> str | None:
        """Extract a time period string from a text snippet."""
        match = self.PERIOD_PATTERN.search(text)
        return match.group(1) if match else None
