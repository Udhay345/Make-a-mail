"""Search query planning, Tavily web research, and grounded contact/email extraction."""

import json
import re
from typing import Any

from llm_helper import call_llm_json

MAX_QUERIES = 3
MAX_RESULTS_PER_QUERY = 3
MAX_SNIPPET_CHARS = 800
QUERY_PLAN_MAX_TOKENS = 350
CONTACT_EXTRACT_MAX_TOKENS = 700

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IGNORED_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "domain.com",
    "email.com",
    "sentry.io",
    "wixpress.com",
    "schema.org",
    "w3.org",
    "creativecommons.org",
    "linkedin.com",
}
_IGNORED_EMAIL_LOCAL_PREFIXES = (
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "support",
    "privacy",
    "abuse",
)
_IGNORED_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js", "woff", "ttf"}

_QUERY_SYSTEM_PROMPT = """You are a research query planner for cold outreach.
Given a candidate's structured resume and outreach intent (Internship, Research position, or Job),
generate ONLY a JSON array of 3 targeted web search queries to discover matching organizations
AND publicly listed HR, recruiter, or professor email addresses.

Rules:
- If a specific target is provided, focus the queries on that target.
- If no target is provided, infer the best-matching technical domain, industry, or research area from the candidate's resume.
- Design the 3 queries across complementary public sources where real email addresses appear:
  1. Company careers / recruiting / HR contact pages in the candidate's domain (e.g., "<domain> careers recruiting hr email").
  2. LinkedIn hiring posts where recruiters share contact emails (e.g., "site:linkedin.com/posts <domain> hiring send resume email").
  3. University research labs / faculty pages with public professor emails (e.g., "<domain> research lab professor faculty email .edu").
- Keep each query concise (under 14 words).
- Return ONLY a JSON array of 3 strings, e.g. ["query 1", "query 2", "query 3"].
- Do not include explanations or markdown."""

_CONTACT_SYSTEM_PROMPT = """You extract verified outreach contacts from web search results.
Given a candidate's resume, outreach intent, and web search snippets (with any detected emails),
return ONLY a JSON array of relevant HR, recruiter, or professor contacts found in the sources.

Each element MUST follow this schema:
{
  "name": "",
  "role": "",
  "organization": "",
  "email": "",
  "reason": "",
  "source_url": ""
}

Strict rules:
- Use ONLY email addresses that literally appear in the provided sources or detected_emails lists.
- NEVER guess, construct, or invent an email address (such as guessing firstname@company.com).
- If a source has a relevant email address without a person's name, still include it with role (e.g., "HR / Careers" or "Faculty / Lab Contact") and organization inferred strictly from that source.
- Keep "reason" concise (1 sentence explaining why the contact/organization fits the candidate's resume).
- Return ONLY valid JSON array, or [] if no verified emails appear in the sources."""


def extract_emails_from_text(text: str) -> list[str]:
    """Extract deduplicated, plausible contact email addresses from text."""
    if not text:
        return []
    seen: set[str] = set()
    emails: list[str] = []
    for match in _EMAIL_RE.findall(text):
        lower = match.lower()
        local, _, domain = lower.partition("@")
        tld = domain.rsplit(".", 1)[-1]
        if domain in _IGNORED_EMAIL_DOMAINS:
            continue
        if tld in _IGNORED_TLDS:
            continue
        if local.startswith(_IGNORED_EMAIL_LOCAL_PREFIXES):
            continue
        if lower not in seen:
            seen.add(lower)
            emails.append(match)
    return emails


def _build_grounded_snippet(content: str, raw_content: str, emails: list[str]) -> str:
    """Keep snippet under 800 chars, preferring context around detected emails if needed."""
    base = (content or raw_content or "").strip()
    if len(base) <= MAX_SNIPPET_CHARS and (not emails or any(e in base for e in emails)):
        return base[:MAX_SNIPPET_CHARS]

    if emails and raw_content:
        first_email = emails[0]
        pos = raw_content.lower().find(first_email.lower())
        if pos != -1:
            start = max(0, pos - 300)
            end = min(len(raw_content), pos + 500)
            window = raw_content[start:end].strip()
            if window:
                return window[:MAX_SNIPPET_CHARS]

    return base[:MAX_SNIPPET_CHARS]


def _tavily_search_single(tavily_client: Any, query: str) -> dict[str, Any]:
    """Run Tavily search with raw_content enabled, falling back if unsupported."""
    try:
        return tavily_client.search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
            include_raw_content="text",
        )
    except TypeError:
        return tavily_client.search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
        )


def plan_queries(
    client: Any,
    resume_json: dict[str, Any],
    target: str = "",
    intent: str = "Internship",
) -> list[str]:
    """Generate up to 3 focused queries to discover matching organizations and HR/professor emails."""
    clean_target = (target or "").strip()
    user_payload = {
        "target": clean_target or "Discover best-matching companies, institutes, HRs, and professors from resume",
        "intent": (intent or "Internship").strip(),
        "skills": resume_json.get("skills", []),
        "projects": resume_json.get("projects", []),
        "education": resume_json.get("education", []),
        "experience": resume_json.get("experience", []),
    }
    messages = [
        {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Candidate & outreach context:\n{json.dumps(user_payload, ensure_ascii=False)}",
        },
    ]

    raw = call_llm_json(client, messages, max_tokens=QUERY_PLAN_MAX_TOKENS, expected_type=list)
    if isinstance(raw, dict):
        for key in ("queries", "search_queries", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("Query planner did not return a JSON list.")

    queries = [str(q).strip() for q in raw if str(q).strip()][:MAX_QUERIES]
    if not queries:
        raise ValueError("Query planner returned no usable search queries.")

    return queries


def search(
    tavily_client: Any,
    target: str = "",
    queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run Tavily searches (max 3 results/query), deduplicate by URL, and extract emails."""
    clean_target = (target or "").strip()
    seen_urls: set[str] = set()
    findings: list[dict[str, Any]] = []

    for query in (queries or [])[:MAX_QUERIES]:
        effective_query = (
            query
            if not clean_target or clean_target.lower() in query.lower()
            else f"{clean_target} {query}"
        )
        response = _tavily_search_single(tavily_client, effective_query)
        for item in response.get("results", []) or []:
            url = str(item.get("url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = str(item.get("title", "") or url).strip()
            content = str(item.get("content", "") or "").strip()
            raw_content = str(item.get("raw_content", "") or "").strip()

            emails = extract_emails_from_text(f"{title}\n{content}\n{raw_content}")
            snippet = _build_grounded_snippet(content, raw_content, emails)
            findings.append(
                {
                    "url": url,
                    "title": title,
                    "text": snippet,
                    "emails": emails,
                }
            )

    return findings


def extract_contacts(
    client: Any,
    resume_json: dict[str, Any],
    sources: list[dict[str, Any]],
    intent: str = "Internship",
) -> list[dict[str, str]]:
    """Extract and strictly verify HR/professor email contacts from retrieved web sources."""
    if not sources:
        return []

    verified_email_sources: dict[str, dict[str, str]] = {}
    for src in sources:
        url = str(src.get("url", "") or "").strip()
        title = str(src.get("title", "") or url).strip()
        emails = list(src.get("emails") or []) + extract_emails_from_text(src.get("text", ""))
        for email in emails:
            key = email.lower()
            if key not in verified_email_sources:
                verified_email_sources[key] = {
                    "email": email,
                    "title": title,
                    "url": url,
                }

    if not verified_email_sources:
        return []

    sources_with_emails = [
        src
        for src in sources
        if src.get("emails") or extract_emails_from_text(src.get("text", ""))
    ][:6]

    payload = {
        "intent": (intent or "Internship").strip(),
        "skills": (resume_json.get("skills") or [])[:15],
        "sources": [
            {
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "text": str(src.get("text", "") or "")[:300],
                "detected_emails": src.get("emails", []),
            }
            for src in sources_with_emails
        ],
    }
    messages = [
        {"role": "system", "content": _CONTACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract verified contacts from these sources:\n{json.dumps(payload, ensure_ascii=False)}",
        },
    ]

    contacts: list[dict[str, str]] = []
    seen_emails: set[str] = set()

    try:
        raw = call_llm_json(client, messages, max_tokens=CONTACT_EXTRACT_MAX_TOKENS, expected_type=list)
    except ValueError:
        raw = []

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            email = str(entry.get("email", "") or "").strip()
            key = email.lower()
            if not email or key not in verified_email_sources or key in seen_emails:
                continue
            seen_emails.add(key)
            fallback_src = verified_email_sources[key]
            contacts.append(
                {
                    "name": str(entry.get("name", "") or "").strip(),
                    "role": str(entry.get("role", "") or "Contact").strip(),
                    "organization": str(entry.get("organization", "") or fallback_src["title"]).strip(),
                    "email": fallback_src["email"],
                    "reason": str(entry.get("reason", "") or "").strip(),
                    "source_url": str(entry.get("source_url", "") or fallback_src["url"]).strip(),
                }
            )

    # Include any regex-detected verified emails from sources that the LLM omitted.
    for key, info in verified_email_sources.items():
        if key in seen_emails:
            continue
        seen_emails.add(key)
        contacts.append(
            {
                "name": "",
                "role": "HR / Faculty Contact",
                "organization": info["title"],
                "email": info["email"],
                "reason": "Public contact email found in search results.",
                "source_url": info["url"],
            }
        )

    return contacts
