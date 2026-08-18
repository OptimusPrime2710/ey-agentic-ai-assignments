"""Generate and conservatively fact-check a two-paragraph client account review.

Installation:
    pip install openai

Configuration (PowerShell):
    $env:OPENROUTER_API_KEY = "your-key"
    $env:OPENROUTER_MODEL = "your-openrouter-model"  # optional; a default is provided

Run:
    python client_account_review.py
    python client_account_review.py path\to\activity.txt
    python client_account_review.py --activity-file path\to\activity.txt

The program uses two independent OpenRouter calls: one to draft the review and one
 to extract and verify claims. It fails closed whenever validation is incomplete.
"""

from __future__ import annotations

import json
import os
import re
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_ACTIVITY = """ACCOUNT ACTIVITY EXTRACT
Client name: Jordan Lee
Statement period: 2026-07-01 through 2026-07-31
Account: Premier Checking (ending 4821)
Opening balance: $12,480.55
Closing balance: $9,875.20
Total inflows: $8,200.00
Total outflows: $10,805.35
Transactions:
2026-07-02 | Payroll deposit | +$4,100.00 | Premier Checking
2026-07-08 | Transfer from savings | +$2,500.00 | Premier Checking
2026-07-11 | Rent payment | -$2,250.00 | Premier Checking
2026-07-15 | Card purchase at Northstar Travel | -$3,450.00 | Premier Checking
2026-07-18 | Utility payment | -$185.35 | Premier Checking
2026-07-22 | Payroll deposit | +$1,600.00 | Premier Checking
2026-07-25 | Monthly maintenance fee | -$12.00 | Premier Checking
2026-07-29 | Credit card payment missed | $0.00 | Premier Checking
Fees: $12.00 monthly maintenance fee charged on 2026-07-25
Product usage: Premier Checking; savings transfer received on 2026-07-08
Sensitive item: $3,450.00 card debit at Northstar Travel on 2026-07-15; a credit card payment was missed on 2026-07-29.
"""

WRITING_SYSTEM_PROMPT = """You are a relationship manager writing directly to the named client.
Use ONLY figures, dates, products, balances, transactions, fees, and events explicitly
present in the supplied source. Never invent, infer, estimate, round, calculate, or
substitute factual details. Produce exactly two prose paragraphs, separated by exactly
one blank line. Do not use bullets, headings, tables, or reprint the statement. Do not
recommend trades or specific investment actions, promise or imply returns, or invent
next-best-product offers or other products/services. Handle sensitive items neutrally
and accurately. If the source is incomplete or insufficient to support a statement,
explicitly say what is missing rather than filling the gap. Keep the tone professional,
concise, client-appropriate, and useful. Preserve the exact factual meaning of source
figures and events. Return only the two paragraphs, with no preamble or commentary."""

FACT_CHECK_SYSTEM_PROMPT = """You are a strict source-grounded fact checker. Compare the draft only
against the supplied source. Extract every explicit number, percentage, date, account
balance, monetary amount, transaction amount, fee, count, named product/account, and
named event. Do not assume a fact is supported merely because it is plausible or can
be calculated from other source values. A claim is IN SOURCE only when its exact factual
meaning is explicitly present in the source; equivalent formatting of the same value is
allowed. Calculations, interpretations, causal claims, sentiment, and recommendations
are NOT IN SOURCE unless explicitly stated. Return valid JSON only in this shape:
{"facts":[{"fact":"exact claim copied from draft","status":"IN SOURCE"|"NOT IN SOURCE","reason":"short explanation"}]}
Include every relevant claim, including sensitive events. Do not omit unsupported claims."""

PROHIBITED_PATTERNS = (
    r"\b(?:buy|sell|trade|invest|switch|purchase)\b",
    r"\b(?:guarantee|guaranteed|will earn|will grow|returns?|profit|yield)\b",
    r"\b(?:recommend|recommendation|should consider)\b",
)


@dataclass(frozen=True)
class Fact:
    fact: str
    status: str
    reason: str


def require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Set it in the environment and rerun."
        )
    return api_key


def load_activity(file_path: str | None) -> str:
    """Load activity from a UTF-8 text file, or use the embedded demonstration."""
    if file_path is None:
        return SAMPLE_ACTIVITY

    path = Path(file_path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"Activity file was not found: {path}")
    try:
        activity = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read activity file '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Activity file must be UTF-8 text: {path}") from exc
    if not activity:
        raise RuntimeError(f"Activity file is empty: {path}")
    return activity


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and fact-check a two-paragraph client account review."
    )
    parser.add_argument(
        "activity_file",
        nargs="?",
        help="Optional UTF-8 text file containing account activity. Uses the embedded sample when omitted.",
    )
    parser.add_argument(
        "--activity-file",
        dest="activity_file_option",
        help="Optional UTF-8 text file containing account activity.",
    )
    args = parser.parse_args()
    if args.activity_file and args.activity_file_option:
        parser.error("provide the activity file either positionally or with --activity-file, not both")
    args.activity_file = args.activity_file_option or args.activity_file
    return args


def call_llm(client: Any, system_prompt: str, user_content: str) -> str:
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The LLM returned an empty response.")
    return content.strip()


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def has_prohibited_content(text: str) -> bool:
    print("Checking for prohibited content in the draft...")
    print([re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROHIBITED_PATTERNS])
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROHIBITED_PATTERNS)


def parse_fact_check(raw: str) -> list[Fact]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        parsed: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fact-check response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        raise ValueError("Fact-check response does not contain a facts list.")

    facts: list[Fact] = []
    for item in parsed["facts"]:
        if not isinstance(item, dict):
            raise ValueError("Fact-check contains a malformed fact item.")
        fact, status, reason = item.get("fact"), item.get("status"), item.get("reason")
        if not all(isinstance(value, str) and value.strip() for value in (fact, status, reason)):
            raise ValueError("Each fact must include non-empty fact, status, and reason fields.")
        normalized_status = status.strip().upper()
        if normalized_status not in {"IN SOURCE", "NOT IN SOURCE"}:
            raise ValueError(f"Unknown fact-check status: {status}")
        facts.append(Fact(fact.strip(), normalized_status, reason.strip()))
    return facts


def render_fact_check(facts: list[Fact]) -> None:
    if not facts:
        print("No facts were returned; conservative validation requires a manual review.")
        return
    for index, fact in enumerate(facts, start=1):
        print(f"{index}. {fact.fact} — {fact.status} ({fact.reason})")


def main() -> int:
    try:
        arguments = parse_arguments()
        activity = load_activity(arguments.activity_file)
        api_key = require_api_key()
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("The openai package is missing. Install it with: pip install openai") from exc
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Client Account Review"},
        )
        draft = call_llm(
            client,
            WRITING_SYSTEM_PROMPT,
            f"SOURCE ACCOUNT ACTIVITY:\n{activity}",
        )
        fact_check_raw = call_llm(
            client,
            FACT_CHECK_SYSTEM_PROMPT,
            f"SOURCE ACCOUNT ACTIVITY:\n{activity}\n\nDRAFT REVIEW:\n{draft}",
        )
        facts = parse_fact_check(fact_check_raw)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    exactly_two_paragraphs = paragraph_count(draft) == 2
    no_prohibited_content = not has_prohibited_content(draft)
    all_supported = bool(facts) and all(fact.status == "IN SOURCE" for fact in facts)
    reasons: list[str] = []
    if not exactly_two_paragraphs:
        reasons.append("the review does not contain exactly two paragraphs")
    if not all_supported:
        reasons.append("the fact check found an unsupported or missing fact")
    if not no_prohibited_content:
        reasons.append("the review contains a prohibited recommendation or return promise")

    print("\n=== CLIENT REVIEW ===")
    print(draft)
    print("\n=== FACT CHECK ===")
    render_fact_check(facts)
    print("\n=== SEND DECISION ===")
    if not reasons:
        print("SEND AS-IS")
    else:
        print(f"DO NOT SEND AS-IS — {'; '.join(reasons)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
