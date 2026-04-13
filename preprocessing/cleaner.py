import re


class TextCleaner:
    """Clean financial text by removing HTML artifacts, normalizing figures,
    percentages, boilerplate, and whitespace."""

    HTML_ENTITY_MAP = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&nbsp;": " ",
        "&quot;": '"',
        "&#39;": "'",
    }

    BOILERPLATE_PATTERNS = [
        "this report contains forward-looking statements",
        "safe harbor statement",
        "pursuant to the securities exchange act",
        "this information is incorporated by reference",
        "as filed with the securities and exchange commission",
        "the company undertakes no obligation to update",
        "certain statements in this",
        "within the meaning of section 27a",
        "within the meaning of the private securities litigation",
        "actual results may differ materially",
        "see risk factors for a discussion",
        "all rights reserved",
    ]

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline in the exact required order."""
        text = self.strip_html_artifacts(text)
        text = self.normalize_financial_figures(text)
        text = self.normalize_percentages(text)
        text = self.remove_boilerplate(text)
        text = self.normalize_whitespace(text)
        return text

    def strip_html_artifacts(self, text: str) -> str:
        """Remove residual HTML tags and convert common HTML entities."""
        text_no_tags = re.sub(r"<[^>]+>", " ", text)
        for entity, replacement in self.HTML_ENTITY_MAP.items():
            text_no_tags = text_no_tags.replace(entity, replacement)
        return text_no_tags

    def normalize_financial_figures(self, text: str) -> str:
        """Normalize large financial figures to compact unit suffixes."""
        def format_number(value: float) -> str:
            if value.is_integer():
                return f"{int(value)}"
            return f"{value:.1f}"

        def replace_scale(match: re.Match) -> str:
            number = float(match.group("num"))
            unit = match.group("unit").lower()
            suffix = ""
            if "billion" in unit:
                suffix = "B"
            elif "million" in unit:
                suffix = "M"
            elif "trillion" in unit:
                suffix = "T"
            if suffix:
                return f"${format_number(number)}{suffix}"
            return match.group(0)

        text = re.sub(
            r"\$?(?P<num>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>billion|million|trillion) dollars?",
            replace_scale,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\$?(?P<num>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>billion|million|trillion)",
            replace_scale,
            text,
            flags=re.IGNORECASE,
        )
        return text

    def normalize_percentages(self, text: str) -> str:
        """Normalize percentage expressions to percent and percentage point forms."""
        def format_number(value: float) -> str:
            if value.is_integer():
                return f"{int(value)}"
            return f"{value:.1f}"

        text = re.sub(
            r"(?P<num>[0-9]+(?:\.[0-9]+)?)\s*percentage points?",
            lambda match: f"{format_number(float(match.group('num')))} pp",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?P<num>[0-9]+(?:\.[0-9]+)?)\s*per cent",
            lambda match: f"{format_number(float(match.group('num')))}%",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?P<num>[0-9]+(?:\.[0-9]+)?)\s*percent",
            lambda match: f"{format_number(float(match.group('num')))}%",
            text,
            flags=re.IGNORECASE,
        )
        return text

    def remove_boilerplate(self, text: str) -> str:
        """Remove full sentences that contain exact SEC/legal boilerplate phrases."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        cleaned_sentences: list[str] = []
        for sentence in sentences:
            normalized_sentence = sentence.lower()
            if any(pattern in normalized_sentence for pattern in self.BOILERPLATE_PATTERNS):
                continue
            cleaned_sentences.append(sentence)
        return " ".join(cleaned_sentences).strip()

    def normalize_whitespace(self, text: str) -> str:
        """Collapse repeated spaces and newlines and strip surrounding whitespace."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
