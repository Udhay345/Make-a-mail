"""Streamlit UI for Make a Mail: automated HR/professor discovery & grounded drafting."""

from pathlib import Path
from urllib.parse import quote

from groq import Groq
import streamlit as st
from tavily import TavilyClient

from email_writer import draft_email
from research import extract_contacts, plan_queries, search
from resume_parser import extract_text, parse_resume

DEMO_RUN_LIMIT = 3
INTENT_OPTIONS = ["Internship", "Research position", "Job"]
MANUAL_EMAIL_OPTION = "Enter email manually..."
LOGO_BG_REMOVED_PATH = Path(__file__).resolve().parent / "assets" / "logo(bg-removed).png"


def _get_secret(key: str) -> str:
    """Safely read a secret from st.secrets without failing if secrets.toml is absent."""
    try:
        return str(st.secrets.get(key, "") or "").strip()
    except Exception:
        return ""


def _resolve_api_keys() -> tuple[str, str, bool]:
    """Return (groq_key, tavily_key, is_demo_mode) from session_state or st.secrets."""
    user_groq = str(st.session_state.get("user_groq_key", "") or "").strip()
    user_tavily = str(st.session_state.get("user_tavily_key", "") or "").strip()

    if user_groq and user_tavily:
        return user_groq, user_tavily, False

    demo_groq = _get_secret("GROQ_API_KEY")
    demo_tavily = _get_secret("TAVILY_API_KEY")

    if not user_groq and not user_tavily and demo_groq and demo_tavily:
        return demo_groq, demo_tavily, True

    return user_groq, user_tavily, False


def _friendly_error(exc: Exception) -> str:
    """Map raw exceptions to user-friendly messages without exposing secrets."""
    msg = str(exc).lower()
    if "401" in msg or "invalid api key" in msg or "unauthorized" in msg or "authentication" in msg:
        return "Authentication failed. Please double-check your Groq and Tavily API keys."
    if "413" in msg or "429" in msg or "rate_limit" in msg or "rate limit" in msg or "tokens per minute" in msg:
        return (
            "Groq token-per-minute limit (8,000 TPM) reached. "
            "Please wait ~30-60 seconds and try again."
        )
    if "timeout" in msg or "connection" in msg:
        return "Network timeout while contacting an external service. Please try again."
    return f"Could not complete the workflow: {exc}"


def _build_mailto_url(to_email: str, subject: str, body: str) -> str:
    """Build a properly percent-encoded mailto: URL for the selected recipient."""
    recipient = quote((to_email or "").strip(), safe="@")
    encoded_subject = quote(subject or "", safe="")
    encoded_body = quote(body or "", safe="")
    return f"mailto:{recipient}?subject={encoded_subject}&body={encoded_body}"


def _build_gmail_url(to_email: str, subject: str, body: str) -> str:
    """Build a direct Gmail web compose URL pre-filled with recipient, subject, and body."""
    recipient = quote((to_email or "").strip(), safe="@")
    encoded_subject = quote(subject or "", safe="")
    encoded_body = quote(body or "", safe="")
    return f"https://mail.google.com/mail/?view=cm&fs=1&to={recipient}&su={encoded_subject}&body={encoded_body}"


def _format_contact_option(contact: dict[str, str]) -> str:
    """Format a discovered HR/professor contact concisely for the dropdown."""
    email = contact.get("email", "")
    name = contact.get("name", "")
    role = contact.get("role", "")
    org = contact.get("organization", "")
    details = ", ".join(part for part in (name, role, org) if part)
    return f"{email} — {details}" if details else email


def _render_sidebar() -> tuple[str, str, bool, bool, bool]:
    """Render sidebar API inputs + Generate button and return state flags."""
    if "demo_runs" not in st.session_state:
        st.session_state["demo_runs"] = 0

    with st.sidebar:
        st.header("API Configuration")
        st.caption("Keys stay in your browser session only and are never saved.")
        st.text_input(
            "Groq API Key",
            type="password",
            key="user_groq_key",
            placeholder="gsk_...",
        )
        st.text_input(
            "Tavily API Key",
            type="password",
            key="user_tavily_key",
            placeholder="tvly-...",
        )

        sidebar_generate = st.button("Generate", type="primary", use_container_width=True)

        groq_key, tavily_key, is_demo = _resolve_api_keys()
        demo_exhausted = False

        if sidebar_generate:
            if groq_key and tavily_key:
                st.session_state["api_ready"] = True
                if st.session_state.get("uploaded_resume") is None:
                    st.success("API keys ready! Upload your resume on the right to continue.")
            else:
                st.warning("Please enter both your Groq API key and Tavily API key.")

        if is_demo:
            used = st.session_state["demo_runs"]
            remaining = max(0, DEMO_RUN_LIMIT - used)
            st.info(f"Demo mode active: {remaining} of {DEMO_RUN_LIMIT} runs remaining.")
            if used >= DEMO_RUN_LIMIT:
                demo_exhausted = True
                st.warning("Add your own key to continue.")
        elif not (groq_key and tavily_key):
            st.info("Enter both API keys above and click Generate to start.")

        st.divider()
        st.caption(
            "Human-in-the-loop: Make a Mail finds verified contacts and drafts emails "
            "for your review. Nothing is ever sent automatically."
        )

    return groq_key, tavily_key, is_demo, demo_exhausted, sidebar_generate


def _run_pipeline(
    groq_key: str,
    tavily_key: str,
    uploaded_file,
    intent: str,
    is_demo: bool,
) -> None:
    """Execute resume parsing, web/LinkedIn contact search, and grounded drafting."""
    if is_demo:
        st.session_state["demo_runs"] = st.session_state.get("demo_runs", 0) + 1

    try:
        groq_client = Groq(api_key=groq_key)
        tavily_client = TavilyClient(api_key=tavily_key)

        with st.status("Running Make a Mail...", expanded=True) as status:
            status.write("1/5 Extracting and parsing resume...")
            resume_text = extract_text(uploaded_file)
            resume_json = parse_resume(groq_client, resume_text)

            status.write("2/5 Planning web and LinkedIn discovery queries...")
            queries = plan_queries(groq_client, resume_json, target="", intent=intent)

            status.write("3/5 Searching web & LinkedIn for matching organizations, HRs, and professors...")
            sources = search(tavily_client, target="", queries=queries)

            status.write("4/5 Verifying HR and professor email addresses from sources...")
            contacts = extract_contacts(groq_client, resume_json, sources, intent=intent)

            status.write("5/5 Drafting grounded outreach email...")
            draft = draft_email(
                groq_client,
                resume_json,
                sources,
                target="",
                intent=intent,
                contacts=contacts,
            )

            status.update(
                label="Discovery and email draft ready for review!",
                state="complete",
                expanded=False,
            )

        st.session_state["results"] = {
            "resume_json": resume_json,
            "queries": queries,
            "sources": sources,
            "contacts": contacts,
            "draft": draft,
        }
        st.session_state["draft_subject"] = draft.get("subject", "")
        st.session_state["draft_body"] = draft.get("body", "")
        st.session_state.pop("selected_recipient_option", None)
    except Exception as exc:
        st.error(_friendly_error(exc))


def _render_resume_and_sources(results: dict) -> None:
    """Render collapsible summaries of the parsed resume and web research findings."""
    resume_json = results["resume_json"]
    with st.expander("Parsed Resume Summary", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Name:** {resume_json.get('name') or 'Not specified'}")
            st.markdown(f"**Email:** {resume_json.get('email') or 'Not specified'}")
            education = resume_json.get("education") or []
            st.markdown("**Education:**")
            if education:
                for item in education:
                    st.markdown(f"- {item}")
            else:
                st.caption("None listed")

        with col_b:
            skills = resume_json.get("skills") or []
            st.markdown("**Skills:**")
            st.write(", ".join(skills) if skills else "None listed")

            experience = resume_json.get("experience") or []
            st.markdown("**Experience:**")
            if experience:
                for item in experience:
                    st.markdown(f"- {item}")
            else:
                st.caption("None listed")

        projects = resume_json.get("projects") or []
        if projects:
            st.markdown("**Projects:**")
            for proj in projects:
                title = proj.get("title") or "Untitled Project"
                summary = proj.get("summary") or ""
                st.markdown(f"- **{title}**: {summary}" if summary else f"- **{title}**")

    with st.expander("Web & LinkedIn Research Sources", expanded=False):
        queries = results.get("queries") or []
        if queries:
            st.markdown("**Search Queries Used:** " + " | ".join(f"`{q}`" for q in queries))

        sources = results.get("sources") or []
        if not sources:
            st.warning(
                "No relevant web results were found. "
                "The draft avoids inventing organization-specific details."
            )
            return

        for idx, src in enumerate(sources, start=1):
            title = src.get("title") or src.get("url")
            url = src.get("url", "")
            snippet = src.get("text", "")
            found_emails = src.get("emails") or []
            st.markdown(f"**{idx}. [{title}]({url})**")
            if found_emails:
                st.caption("Emails detected: " + ", ".join(f"`{e}`" for e in found_emails))
            if snippet:
                st.caption(snippet)


def _render_review_screen(results: dict) -> None:
    """Render mail body on the left, discovered HR/professor email dropdown on the right, and bottom email button."""
    st.divider()
    st.subheader("2. Discovered Contacts & Outreach Draft")

    _render_resume_and_sources(results)

    contacts = results.get("contacts") or []
    draft = results["draft"]

    # Mail body on the left; recipient email dropdown + discovered HR/professor list on the right.
    mail_col, contacts_col = st.columns([3, 2], gap="large")

    with contacts_col:
        st.markdown("#### Select HR / Professor Email")
        options_map: dict[str, dict[str, str]] = {}
        dropdown_options: list[str] = []
        for contact in contacts:
            label = _format_contact_option(contact)
            if label not in options_map:
                options_map[label] = contact
                dropdown_options.append(label)
        dropdown_options.append(MANUAL_EMAIL_OPTION)

        selected_label = st.selectbox(
            "Recipient Email Address",
            options=dropdown_options,
            key="selected_recipient_option",
        )

        if selected_label == MANUAL_EMAIL_OPTION:
            selected_email = st.text_input(
                "Enter Recipient Email",
                placeholder="recruiter@company.com or professor@university.edu",
                key="manual_recipient_email",
            ).strip()
            selected_contact = None
        else:
            selected_contact = options_map[selected_label]
            selected_email = selected_contact["email"]
            if selected_contact.get("reason"):
                st.caption(f"Why it fits: {selected_contact['reason']}")
            if selected_contact.get("source_url"):
                st.markdown(f"Source: [{selected_contact['source_url']}]({selected_contact['source_url']})")

        st.markdown("#### Discovered HR & Professor Emails")
        if contacts:
            for c in contacts:
                header_parts = [p for p in (c.get("name"), c.get("role"), c.get("organization")) if p]
                header = " — ".join(header_parts) if header_parts else "Discovered Contact"
                st.markdown(f"- **`{c['email']}`** ({header})")
                if c.get("source_url"):
                    st.caption(f"Source: {c['source_url']}")
        else:
            st.info(
                "No public email addresses were directly present in the retrieved snippets. "
                "Check the research sources above or enter an email manually."
            )

        skill_matches = draft.get("skill_matches") or []
        if skill_matches:
            st.markdown("#### Grounded Skill Matches")
            for match in skill_matches:
                skill = match.get("skill") or "Skill"
                reason = match.get("reason") or ""
                st.markdown(f"- **{skill}**: {reason}")

    with mail_col:
        st.markdown("#### Email Subject & Body")
        subject = st.text_input("Subject", key="draft_subject")
        body = st.text_area("Mail Body", key="draft_body", height=280)
        word_count = len(body.split())
        if word_count > 200:
            st.caption(f"Word count: {word_count} words (consider trimming under 200 words).")
        else:
            st.caption(f"Word count: {word_count} / 200 words")

        if selected_email:
            st.markdown(f"**Send to:** `{selected_email}`")
        else:
            st.caption("Select a discovered HR/professor email on the right or enter one manually.")

        gmail_url = _build_gmail_url(selected_email, subject, body)
        mailto_url = _build_mailto_url(selected_email, subject, body)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.link_button(
                f"Open in Gmail ({selected_email})" if selected_email else "Open in Gmail",
                url=gmail_url,
                type="primary",
                use_container_width=True,
            )
        with btn_col2:
            st.link_button(
                "Open in Default Email App",
                url=mailto_url,
                use_container_width=True,
            )

        with st.expander("Copy-friendly email text", expanded=False):
            st.code(
                f"To: {selected_email or '(select or enter recipient)'}\n"
                f"Subject: {subject}\n\n{body}",
                language=None,
            )


def main() -> None:
    st.set_page_config(page_title="Make a Mail", page_icon="✉️", layout="wide")

    if LOGO_BG_REMOVED_PATH.exists():
        _, logo_center, _ = st.columns([1.3, 2, 1.3])
        with logo_center:
            st.image(str(LOGO_BG_REMOVED_PATH), width="stretch")

    st.markdown(
        "<p style='text-align: center;'>"
        "Upload your resume to automatically discover matching companies, university institutes, "
        "HR recruiters, and professors—and draft a grounded outreach email for your review."
        "</p>",
        unsafe_allow_html=True,
    )

    groq_key, tavily_key, is_demo, demo_exhausted, sidebar_generate = _render_sidebar()

    if not (groq_key and tavily_key):
        st.info("Enter your Groq and Tavily API keys in the sidebar and click **Generate** to begin.")
        st.stop()

    if is_demo and demo_exhausted and "results" not in st.session_state:
        st.warning("Add your own key to continue.")
        st.stop()

    st.subheader("1. Upload Resume")
    col_upload, col_intent = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload your resume (.pdf or .txt)",
            type=["pdf", "txt"],
            key="uploaded_resume",
        )
    with col_intent:
        intent = st.selectbox("Opportunity Type", INTENT_OPTIONS)

    main_generate = st.button(
        "Find Companies, Emails & Draft Mail",
        type="primary",
        disabled=bool(is_demo and demo_exhausted),
    )

    if is_demo and demo_exhausted:
        st.warning("Demo run limit (3 runs) reached. Add your own key to continue.")

    trigger_run = main_generate or (sidebar_generate and uploaded_file is not None)
    if trigger_run:
        if is_demo and st.session_state.get("demo_runs", 0) >= DEMO_RUN_LIMIT:
            st.error("Add your own key to continue.")
        elif uploaded_file is None:
            st.warning("Please upload a PDF or TXT resume first.")
        else:
            _run_pipeline(groq_key, tavily_key, uploaded_file, intent, is_demo)

    if "results" in st.session_state:
        _render_review_screen(st.session_state["results"])

    st.divider()
    st.caption(
        "Privacy notice: Uploaded resumes are processed strictly in memory for the current "
        "session and are not stored on disk or logged."
    )


if __name__ == "__main__":
    main()
