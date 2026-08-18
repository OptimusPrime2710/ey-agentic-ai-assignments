"""Extract customer-email details from a PDF and process them with OpenRouter.

Usage:
    python pdf_customer_email_app.py path\\to\\customer_email.pdf

The PDF must contain selectable text. Scanned PDFs require OCR first.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from customer_email_app import (
    ApplicationError,
    Settings,
    call_openrouter,
    load_settings,
)

MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_TEXT_LENGTH = 100_000


@dataclass(frozen=True)
class SummaryResult:
    """Validated summary data returned for a PDF document."""

    summary: list[str]
    main_issue: str
    important_details: list[str]


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract selectable text from every page without logging its contents."""
    if not pdf_path.exists():
        raise ApplicationError(f"PDF file not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ApplicationError("The supplied PDF path is not a file.")
    if pdf_path.suffix.lower() != ".pdf":
        raise ApplicationError("Please provide a file with a .pdf extension.")
    if pdf_path.stat().st_size > MAX_PDF_SIZE_BYTES:
        raise ApplicationError("The PDF is too large. The maximum supported size is 10 MB.")
    try:
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
    except Exception as exc:
        raise ApplicationError("The PDF could not be read. Confirm that it is a valid, readable PDF.") from exc
    text = "\n\n".join(pages).strip()
    if not text:
        raise ApplicationError("No selectable text was found in the PDF. Use an OCR-enabled PDF for scanned documents.")
    if len(text) > MAX_EXTRACTED_TEXT_LENGTH:
        raise ApplicationError(f"The extracted PDF text is too long. The maximum is {MAX_EXTRACTED_TEXT_LENGTH:,} characters.")
    return text


def build_pdf_prompt(extracted_text: str, filename: str) -> str:
    """Build the structured summary prompt for extracted PDF content."""
    return f'''Analyze the customer email details extracted from the PDF named {filename!r}.

Return ONLY one valid JSON object with exactly these fields:
{{
  "summary": ["3 to 5 concise bullet points"],
  "main_issue": "The customer's primary issue or request",
  "important_details": ["deadlines, questions, requested actions, or other important details"]
}}

Rules:
- Use only facts present in the extracted document.
- Do not invent policies, prices, timelines, resolutions, or customer information.
- Preserve names, order numbers, dates, questions, deadlines, and requested actions.
- Use an empty array when no important details are present.
- Return valid JSON without Markdown fences.

BEGIN EXTRACTED PDF TEXT
<pdf_text>
{extracted_text}
</pdf_text>
END EXTRACTED PDF TEXT'''


def parse_summary_result(text: str) -> SummaryResult:
    """Parse and validate summary-only model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ApplicationError("The model response did not contain a JSON object.")
    try:
        data: Any = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ApplicationError("The model returned invalid JSON. Please try again.") from exc
    if not isinstance(data, dict):
        raise ApplicationError("The model response must be a JSON object.")

    summary = data.get("summary")
    main_issue = data.get("main_issue")
    important_details = data.get("important_details")
    if not isinstance(summary, list) or not 3 <= len(summary) <= 5 or not all(
        isinstance(item, str) and item.strip() for item in summary
    ):
        raise ApplicationError("The model returned an invalid summary. Expected 3–5 text points.")
    if not isinstance(main_issue, str) or not main_issue.strip():
        raise ApplicationError("The model returned an invalid main issue.")
    if not isinstance(important_details, list) or not all(
        isinstance(item, str) and item.strip() for item in important_details
    ):
        raise ApplicationError("The model returned invalid important details.")
    return SummaryResult(
        [item.strip() for item in summary],
        main_issue.strip(),
        [item.strip() for item in important_details],
    )


def display_summary(result: SummaryResult) -> None:
    """Display only the summary information for a PDF document."""
    print("\n" + "=" * 72)
    print("DOCUMENT SUMMARY\n" + "=" * 72)
    for point in result.summary:
        print(f"• {point}")
    print(f"\nMAIN ISSUE\n{result.main_issue}\n\nIMPORTANT DETAILS")
    if result.important_details:
        for detail in result.important_details:
            print(f"• {detail}")
    else:
        print("None identified.")
    print("=" * 72)


def process_pdf(pdf_path: Path, settings: Settings) -> SummaryResult:
    """Extract the PDF, call OpenRouter, and validate the result."""
    extracted_text = extract_pdf_text(pdf_path)
    response_text = call_openrouter(settings, build_pdf_prompt(extracted_text, pdf_path.name))
    return parse_summary_result(response_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Process customer email details from a PDF with OpenRouter.")
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF containing the customer email")
    try:
        args = parser.parse_args()
        result = process_pdf(args.pdf, load_settings())
        display_summary(result)
        return 0
    except (ApplicationError, OSError, KeyboardInterrupt) as exc:
        print(f"Error: {'Input cancelled.' if isinstance(exc, KeyboardInterrupt) else exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
