"""Process a customer email with an OpenRouter language model."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_RETRIES = 2
MAX_EMAIL_LENGTH = 50_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class ApplicationError(Exception):
    """An expected, user-facing application error."""


@dataclass(frozen=True)
class CustomerEmail:
    address: str
    subject: str
    body: str


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    api_url: str
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class ProcessingResult:
    summary: list[str]
    main_issue: str
    important_details: list[str]
    customer_reply: str


def load_settings() -> Settings:
    """Load and validate settings without printing the API key."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ApplicationError(
            "OPENROUTER_API_KEY is not configured. Add it to .env or set it as an environment variable."
        )
    try:
        timeout = float(os.getenv("OPENROUTER_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        retries = int(os.getenv("OPENROUTER_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    except ValueError as exc:
        raise ApplicationError(
            "OPENROUTER_TIMEOUT must be a number and OPENROUTER_MAX_RETRIES must be an integer."
        ) from exc
    if not 1 <= timeout <= 300:
        raise ApplicationError("OPENROUTER_TIMEOUT must be between 1 and 300 seconds.")
    if not 0 <= retries <= 5:
        raise ApplicationError("OPENROUTER_MAX_RETRIES must be between 0 and 5.")
    return Settings(
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        api_url=os.getenv("OPENROUTER_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL,
        timeout_seconds=timeout,
        max_retries=retries,
    )


def validate_email(email: CustomerEmail) -> CustomerEmail:
    """Validate and normalize customer email input."""
    address, subject, body = email.address.strip(), email.subject.strip(), email.body.strip()
    if not address or not EMAIL_PATTERN.fullmatch(address):
        raise ApplicationError("Please enter a valid customer email address.")
    if not subject:
        raise ApplicationError("The email subject cannot be empty.")
    if not body:
        raise ApplicationError("The email body cannot be empty.")
    if len(subject) > 500:
        raise ApplicationError("The email subject is too long (maximum: 500 characters).")
    if len(body) > MAX_EMAIL_LENGTH:
        raise ApplicationError(f"The email body is too long (maximum: {MAX_EMAIL_LENGTH:,} characters).")
    return CustomerEmail(address, subject, body)


def build_prompt(email: CustomerEmail) -> str:
    """Build the structured-output prompt."""
    return f'''Analyze the customer email below and return ONLY one valid JSON object.

The JSON must have exactly these fields:
{{
  "summary": ["3 to 5 concise bullet points"],
  "main_issue": "The customer's primary issue or request",
  "important_details": ["deadlines, questions, requested actions, or other important details"],
  "customer_reply": "A complete professional reply ready to copy and send"
}}

Requirements:
- Do not invent policies, prices, timelines, resolutions, or other facts.
- If information is missing, say what information is needed in the reply.
- Use a friendly, professional tone and address the request directly.
- Use empty arrays when there are no important details.
- Return valid JSON without Markdown fences.

CUSTOMER EMAIL ADDRESS:
<customer_address>{email.address}</customer_address>

EMAIL SUBJECT:
<subject>{email.subject}</subject>

EMAIL BODY:
<body>
{email.body}
</body>'''


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def call_openrouter(settings: Settings, prompt: str) -> str:
    """Call OpenRouter and return the assistant content."""
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/customer-email-app",
        "X-Title": "Customer Email Processor",
    }
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": "You are a careful customer-service assistant that follows JSON instructions."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(settings.max_retries + 1):
        try:
            response = requests.post(settings.api_url, headers=headers, json=payload, timeout=settings.timeout_seconds)
        except requests.Timeout as exc:
            if attempt < settings.max_retries:
                time.sleep(2**attempt)
                continue
            raise ApplicationError("The OpenRouter request timed out. Please try again.") from exc
        except requests.RequestException as exc:
            raise ApplicationError("Could not connect to OpenRouter. Check your internet connection.") from exc
        if response.ok:
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ApplicationError("OpenRouter returned an unexpected response format.") from exc
            if not isinstance(content, str) or not content.strip():
                raise ApplicationError("OpenRouter returned an empty response.")
            return content
        if _is_retryable_status(response.status_code) and attempt < settings.max_retries:
            time.sleep(2**attempt)
            continue
        if response.status_code in (401, 403):
            raise ApplicationError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY.")
        if response.status_code == 400:
            raise ApplicationError("OpenRouter rejected the request. Check the configured model and request settings.")
        raise ApplicationError(f"OpenRouter returned HTTP {response.status_code}. Please try again later.")
    raise ApplicationError("The OpenRouter request failed after the configured retries.")


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ApplicationError("The model response did not contain a JSON object.")
    return cleaned[start:end + 1]


def parse_result(text: str) -> ProcessingResult:
    """Parse and validate structured model output."""
    try:
        data: Any = json.loads(_extract_json(text))
    except (json.JSONDecodeError, ApplicationError) as exc:
        raise ApplicationError("The model returned invalid JSON. Please try again.") from exc
    if not isinstance(data, dict):
        raise ApplicationError("The model response must be a JSON object.")
    summary = data.get("summary")
    details = data.get("important_details")
    main_issue = data.get("main_issue")
    reply = data.get("customer_reply")
    if not isinstance(summary, list) or not 3 <= len(summary) <= 5 or not all(isinstance(x, str) and x.strip() for x in summary):
        raise ApplicationError("The model returned an invalid summary. Expected 3–5 text points.")
    if not isinstance(details, list) or not all(isinstance(x, str) and x.strip() for x in details):
        raise ApplicationError("The model returned invalid important details.")
    if not isinstance(main_issue, str) or not main_issue.strip():
        raise ApplicationError("The model returned an invalid main issue.")
    if not isinstance(reply, str) or not reply.strip():
        raise ApplicationError("The model returned an invalid customer reply.")
    return ProcessingResult(summary, main_issue.strip(), details, reply.strip())


def _read_body() -> str:
    print("Enter the full email body. Type a blank line after the body to finish:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def get_customer_email(args: argparse.Namespace) -> CustomerEmail:
    address = args.email or input("Customer email address: ")
    subject = args.subject or input("Email subject: ")
    body = args.body if args.body is not None else _read_body()
    return validate_email(CustomerEmail(address, subject, body))


def display_result(result: ProcessingResult) -> None:
    print("\n" + "=" * 72)
    print("EMAIL SUMMARY\n" + "=" * 72)
    for point in result.summary:
        print(f"• {point}")
    print(f"\nMAIN ISSUE\n{result.main_issue}\n\nIMPORTANT DETAILS")
    if result.important_details:
        for detail in result.important_details:
            print(f"• {detail}")
    else:
        print("None identified.")
    print(f"\nCUSTOMER REPLY\n{result.customer_reply}\n" + "=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a customer email and draft a reply with OpenRouter.")
    parser.add_argument("--email", help="Customer email address")
    parser.add_argument("--subject", help="Email subject")
    parser.add_argument("--body", help="Complete email body")
    try:
        args = parser.parse_args()
        settings = load_settings()
        email = get_customer_email(args)
        result = parse_result(call_openrouter(settings, build_prompt(email)))
        display_result(result)
        return 0
    except (ApplicationError, EOFError, KeyboardInterrupt) as exc:
        print(f"Error: {'Input cancelled.' if isinstance(exc, KeyboardInterrupt) else exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
