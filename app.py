"""
AI Newsletter Generator - Streamlit App
Built using LangChain Agent (Tool Calling) + Gemini + Tavily
"""

import datetime
import re
import uuid
from typing import Optional

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

# To show web-app: complete page layout
st.set_page_config(page_title="AI Newsletter Generator", page_icon="📰", layout="wide")

# Domains must look like real domains (something.tld) for the Tavily API to
# accept them - things like "Government" or "News" are categories, not
# domains, and get silently dropped rather than crashing the whole run.
_DOMAIN_RE = re.compile(
    r"^([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


def _clean_domains(raw_list):
    """Filter a list of user/agent-supplied strings down to ones that look
    like valid domains. Returns (valid, dropped)."""
    valid, dropped = [], []
    for item in raw_list or []:
        item = str(item).strip().lower()
        item = re.sub(r"^https?://", "", item).split("/")[0]  # strip scheme/path if present
        if item and _DOMAIN_RE.match(item):
            valid.append(item)
        elif item:
            dropped.append(item)
    return valid, dropped

# ============ SESSION STATE INIT ===================
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: id, timestamp, title, html, meta
if "last_html" not in st.session_state:
    st.session_state.last_html = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # newsletter_id -> "up"/"down"

# To give title
st.title("AI NEWSLETTER GENERATOR")

st.write("""This app helps you build a curated, styled HTML newsletter
from this week's top trending news using a LangChain agent.""")

st.sidebar.title("Fill Important Details")

# ============ API KEYS ===================
with st.sidebar.expander("🔑 API Keys", expanded=True):
    TAVILY_API_KEY = st.text_input(
        "Tavily-API",
        type="password",
        value=st.secrets.get("TAVILY_API_KEY", "") if hasattr(st, "secrets") else "",
    )
    GOOGLE_API_KEY = st.text_input(
        "Gemini-API",
        type="password",
        value=st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else "",
    )
    remember_keys = st.checkbox(
        "Remember keys for this session", value=True,
        help="Keeps keys in session state so you don't have to re-enter them while you tweak settings.",
    )

all_API = [TAVILY_API_KEY, GOOGLE_API_KEY]

if not all(all_API):
    st.error("Must give API keys")
    st.stop()
elif all(all_API):
    st.sidebar.success("API KEYS LOADED SUCCESSFULLY")

    # =========== MODEL CREATION ==============
    @st.cache_resource(show_spinner=False)
    def get_model(api_key, temperature):
        return ChatGoogleGenerativeAI(
            model='gemini-3.5-flash-lite',
            google_api_key=api_key,
            temperature=temperature,
        )
else:
    st.info("PASS ALL API-KEYS")

# ==================GET USER INFO=====================
st.markdown("### NEWSLETTER DETAILS")
col_title, col_lang = st.columns([2, 1])
with col_title:
    newsletter_title = st.text_input("Newsletter Title", value="Weekly Digest")
with col_lang:
    output_language = st.selectbox(
        "Output Language",
        ["English", "Spanish", "French", "German", "Hindi", "Japanese", "Portuguese", "Arabic"],
        index=0,
    )

# =========== COUNTRY SELECTION ==============
country_options = [
    "United States",
    "United Kingdom",
    "India",
    "Canada",
    "Australia",
    "Germany",
    "France",
    "Japan",
    "China",
    "Custom",
]
col_country, col_category = st.columns(2)
with col_country:
    selected_country = st.selectbox("Country", country_options, index=0)

    if selected_country == "Custom":
        country_choice = st.text_input("Enter Country", value="")
    elif selected_country == "Any":
        country_choice = None
    else:
        country_choice = selected_country

# =========== CATEGORY SELECTION ==============
category_options = [
    "Tech",
    "Business",
    "Science",
    "World",
]
with col_category:
    selected_category = st.selectbox("Category", category_options, index=0)
    category_choice = None if selected_category == "Any" else selected_category

# =========== KEYWORD / TOPIC FOCUS ==============
custom_keywords = st.text_input(
    "Extra keywords / topics to focus on (optional, comma separated)",
    value="",
    help="e.g. 'artificial intelligence, elections, climate' - nudges the search toward these topics.",
)

# =========== ADVANCED SEARCH SETTINGS ==============
with st.sidebar.expander("🔍 Search Settings", expanded=False):
    collector_pool_size = st.slider(
        "Candidate pool size", min_value=5, max_value=30, value=10,
        help="How many candidate articles to fetch before narrowing down to the final 5.",
    )
    time_range = st.selectbox("Time range", ["day", "week", "month", "year"], index=1)
    search_depth = st.selectbox("Search depth", ["basic", "advanced"], index=0)
    include_domains_raw = st.text_input("Only include domains (comma separated)", value="")
    exclude_domains_raw = st.text_input("Exclude domains (comma separated)", value="")

include_domains, _dropped_include = _clean_domains(include_domains_raw.split(","))
exclude_domains, _dropped_exclude = _clean_domains(exclude_domains_raw.split(","))

if _dropped_include:
    st.sidebar.warning(
        f"Ignored invalid include domain(s): {', '.join(_dropped_include)}. "
        "Use a real domain like 'bbc.com', not a category name."
    )
if _dropped_exclude:
    st.sidebar.warning(
        f"Ignored invalid exclude domain(s): {', '.join(_dropped_exclude)}. "
        "Use a real domain like 'bbc.com', not a category name."
    )

# =========== STYLE / TONE SETTINGS ==============
with st.sidebar.expander("🎨 Style Settings", expanded=False):
    tone_choice = st.selectbox(
        "Writing tone", ["Professional", "Casual", "Playful", "Formal", "Enthusiastic"], index=0
    )
    accent_color = st.color_picker("Accent / highlight color", value="#1a73e8")
    creativity = st.slider(
        "Model creativity (temperature)", min_value=0.0, max_value=1.0, value=0.4, step=0.1
    )
    preview_dark_mode = st.checkbox("Preview panel dark background", value=False)

# =========== ARCHIVE / HISTORY ==============
with st.sidebar.expander(f"🗂️ Archive ({len(st.session_state.history)})", expanded=False):
    if not st.session_state.history:
        st.caption("No newsletters generated yet this session.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['title']}** — {item['timestamp']}")
            st.caption(f"{item['meta']['country']} · {item['meta']['category']} · {item['meta']['word_count']} words")
            st.download_button(
                "Download",
                data=item["html"],
                file_name=f"{item['title'].replace(' ', '_')}_{item['id'][:6]}.html",
                mime="text/html",
                key=f"dl_{item['id']}",
            )
            st.divider()
        if st.button("Clear archive"):
            st.session_state.history = []
            st.rerun()


# =========== TOOL 1 ======================
def weekly_article_collector(
    max_results: int = 5,
    country: Optional[str] = None,
    category: Optional[str] = None,
    keywords: Optional[str] = None,
    time_range: str = "week",
    search_depth: str = "basic",
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None,
):
    """This function searches the web for the top trending news
    headlines published recently using the Tavily search API.
    Optionally restricts results to a specific country and/or
    category, nudges the query toward extra keywords/topics, and
    supports a configurable time_range (day/week/month/year),
    search_depth (basic/advanced), and include/exclude domain lists.
    Returns article metadata: title, url, content and published date."""

    query_parts = ["top trending"]
    if category:
        query_parts.append(f"{category}")
    query_parts.append("news headlines")
    if keywords:
        query_parts.append(f"about {keywords}")
    if country:
        query_parts.append(f"in {country}")
    query_parts.append(f"this {time_range}")
    query = " ".join(query_parts)

    # Defensive cleanup: the agent sometimes passes category/keyword-like
    # strings (e.g. "Government") instead of real domains. Silently drop
    # anything that isn't a valid domain rather than letting the API call fail.
    include_domains, _ = _clean_domains(include_domains)
    exclude_domains, _ = _clean_domains(exclude_domains)

    client = TavilyClient(api_key=TAVILY_API_KEY)
    search_kwargs = dict(
        query=query,
        topic="news",
        time_range=time_range,
        max_results=max_results,
        include_answer=False,
        search_depth=search_depth,
    )
    if include_domains:
        search_kwargs["include_domains"] = include_domains
    if exclude_domains:
        search_kwargs["exclude_domains"] = exclude_domains

    response = client.search(**search_kwargs)

    articles = []
    for result in response.get("results", []):
        articles.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "published_date": result.get("published_date", "N/A"),
        })
    return articles


# =========== TOOL 2 ======================
def article_summarizer(article_text, article_title="Untitled", tone="Professional"):
    """This function takes article text or url content and
    produces a concise summary, key points, category
    and relevance score (out of 10) using LLM,
    given article title, content, and a desired writing tone."""

    prompt = f"""You are a professional newsletter editor writing in a {tone.lower()} tone.
    Summarize the article below in 3-4 concise lines,
    then list 2-3 key points as bullets, assign a single
    category (Tech/Business/Science/World/Other) and give
    a relevance score out of 10 for a general tech audience.

    Article Title: {article_title}
    Article Content: {article_text}

    Give output strictly in this format:
    Summary: <summary>
    Key Points: <point1>; <point2>; <point3>
    Category: <category>
    Relevance: <score>/10
    """

    response = model.invoke(prompt)
    return _extract_text(response)


# =========== TOOL 3 ======================
def newsletter_html_generator(
    curated_summaries,
    newsletter_title="Weekly Newsletter",
    style_notes: str = "",
):
    """This function converts curated article summaries
    into a styled html newsletter template suitable
    for email or web publishing, given curated summaries
    text, newsletter title, and optional style_notes (language,
    tone, accent color) that only affect visual/language flavor,
    never the required structure."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")

    prompt = f"""Convert the curated article summaries below into a single
    self-contained HTML page styled like a printed magazine/school
    newsletter front page. Return a full HTML document with a <style>
    block in the <head> (CSS does not need to be inline), max content
    width around 900px, centered on the page with a soft off-white
    background inside a bordered page frame.

    CRITICAL - TEXT MUST ALWAYS BE VISIBLE:
    - Every single text element (body, headings, paragraphs, links, list
      items, spans) MUST have an explicit CSS `color` property with a
      concrete hex/rgb value (e.g. color: #1a1a1a). NEVER leave color
      unset, and never rely on "inherit" or "currentColor".
    - Every colored box, banner, or section background MUST have an
      explicit `background-color` AND an explicit `color` on the text
      inside it, chosen so there is strong contrast (e.g. dark text on
      light backgrounds, or white text on dark backgrounds). Never pair
      a light background with unset/light text or a dark background with
      unset/dark text.
    - Set `color` and `background-color` explicitly on the <body> and
      <html> tags too, so nothing depends on browser or host-page
      defaults.

    Do NOT include any <img> tags, background-image styles, or image
    urls anywhere in the output - text and colored boxes only, no photos.

    Follow this exact structure and style:
    - Top-left corner: a small bordered box showing the exact text
      "{current_date}" as the generation/publish date.
    - Masthead: one huge bold centered title showing
      "{newsletter_title}" (60-90px, wide letter-spacing, like a
      newspaper nameplate).
    - Sub-banner: a full-width colored horizontal band directly below
      the masthead with small bold centered uppercase tagline text
      summarizing that this covers this week's top stories across
      any topic/category (do not hardcode a single subject like "AI"
      in the tagline - infer a short general tagline from the actual
      mix of categories present in the curated summaries below).
    - Below the banner, use a CSS grid with exactly 2 columns
      (display: grid; grid-template-columns: 1fr 1fr; column-gap and
      row-gap around 24px) to lay out one section per curated article.
      Do NOT create a single "top story" block - turn every curated
      article into its own section: a bold uppercase heading (short,
      based on the article's own title/category) followed by a
      paragraph using that article's summary and key points as body
      text, spanning whatever categories the curated summaries
      actually cover (world news, tech, business, science, etc).
    - CRITICAL - there will always be EXACTLY 5 curated articles below.
      Render EXACTLY 5 sections in the grid, one per article - never
      more, never fewer. Since 5 is odd, the LAST article's section
      must span both columns (grid-column: 1 / -1) instead of leaving
      an empty cell next to it. Never render an empty box, an empty
      <div>, or a solid-colored block with no text in it anywhere on
      the page - every colored box must contain real content from the
      curated summaries.
    - For 2-3 of the sections, add a short colored accent line
      (in a highlight color) above the paragraph showing that article's
      own category or relevance score.
    - At the bottom of each column, add one pale colored info box
      (light blue/light pink) containing that source article's category
      and a bold "Read More" link pointing to the real article url -
      only add this info box if there is a real article category and
      url to put inside it, never as an empty filler box.
      Use the real curated article titles, summaries, categories and
      urls throughout - never placeholder/lorem ipsum text.
    - Footer: a full-width colored strip at the very bottom with small
      bold centered white text reading something like "Compiled
      automatically by a multi-agent AI pipeline | Generated on
      {current_date}".
    Give final response strictly in HTML only, no markdown, no code fences.

    Newsletter Title: {newsletter_title}
    Generated On: {current_date}
    Curated Summaries (multiple topics/categories from this week): {curated_summaries}
    """

    if style_notes:
        prompt += (
            "\n\nAdditional style preference (do NOT change the structure, "
            "sections, or counts described above - only apply this as a "
            f"visual/language flavor on top of it): {style_notes}"
        )

    response = model.invoke(prompt)
    return _extract_text(response)


def _extract_text(response):
    """Safely extract text from a LangChain chat model response, regardless
    of whether .content is a plain string or a list of content blocks."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict) and "text" in last:
            return last["text"]
        return str(last)
    return str(content)


def _strip_html_to_text(html):
    """Rough HTML -> plain text conversion for the plain-text/markdown export."""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_links(html):
    """Pull out (text, href) pairs for a quick source list under the preview."""
    return re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL)


def _reading_time_minutes(word_count, wpm=200):
    return max(1, round(word_count / wpm))


max_results = 5  # newsletter template always renders exactly 5 curated sections

if all(all_API):
    model = get_model(GOOGLE_API_KEY, creativity)

    # ========== Agent Creation ================
    agent = create_agent(
        model=model,
        tools=[weekly_article_collector, article_summarizer, newsletter_html_generator]
    )


# ============== MAIN AGENT ===============
def main_agent(agent, query):
    """This is the main agent, or leader agent,
    orchestrates the full newsletter workflow"""

    prompt = """Your task is to orchestrate the full newsletter workflow
    based on the instructions given below:
    1. Call the weekly_article_collector tool with the pool_max_results
       given, and with the country, category, keywords, time_range,
       search_depth, include_domains and exclude_domains as given (if
       any), to fetch a pool of candidate top trending news headlines.
       If a country is given, ONLY include news from that country. If a
       category is given, ONLY include news from that category.
       IMPORTANT: include_domains and exclude_domains must ONLY ever
       contain real website domains (e.g. "bbc.com", "reuters.com").
       NEVER put a category name, keyword, or topic (e.g. "Government",
       "Tech") into include_domains or exclude_domains - if no real
       domains were given, leave those lists empty.
    2. The final newsletter MUST contain EXACTLY 5 curated articles,
       never more and never fewer. If the pool returned fewer than 5
       articles, call weekly_article_collector again with a larger
       pool_max_results (and, if needed, a broader/looser version of
       the country or category constraint) until at least 5 usable
       articles are available.
    3. Call the article_summarizer tool separately on EACH candidate
       article to get its summary, key points, category and relevance
       score, using the requested writing tone.
    4. From the summarized candidates, keep EXACTLY the best 5
       curated articles by relevance score.
    5. Combine those EXACTLY 5 curated summaries (title, summary,
       key points, category, url for each) into one collection, then
       call the newsletter_html_generator tool once with that full
       collection, the newsletter title, and any style_notes (language,
       tone, accent color) so all 5 curated articles appear in the
       final newsletter.
    Give the final response output strictly in HTML, no markdowns,
    no code fences, no explanation text before or after the HTML.
    Instructions given below:
    """

    prompt = prompt + query

    response = agent.invoke({"messages": [{'role': 'user', 'content': prompt}]})
    code = _extract_text(response['messages'][-1])
    return code


# ========== CALLING MAIN AGENT ===============
gen_col, regen_col = st.columns([1, 1])
generate_clicked = gen_col.button("Generate Newsletter", type="primary")
regenerate_clicked = regen_col.button(
    "Regenerate with same settings", disabled=st.session_state.last_meta is None
)

if generate_clicked or regenerate_clicked:
    if country_choice:
        country_line = f"\nCountry: {country_choice} (ONLY use news from this country)."
    else:
        country_line = "\nCountry: Any (do not restrict to a single country)."

    if category_choice:
        category_line = f"\nCategory: {category_choice} (ONLY use news from this category)."
    else:
        category_line = "\nCategory: Any (do not restrict to a single category)."

    keywords_line = f"\nExtra keywords/topics to favor: {custom_keywords}." if custom_keywords else ""
    domains_line = ""
    if include_domains:
        domains_line += f"\nOnly include these domains: {', '.join(include_domains)}."
    if exclude_domains:
        domains_line += f"\nExclude these domains: {', '.join(exclude_domains)}."

    style_notes = (
        f"Write in {output_language}. Use a {tone_choice.lower()} tone throughout. "
        f"Use {accent_color} as the primary accent/highlight color for banners, "
        f"accent lines and info boxes wherever a color choice is needed."
    )

    user_query = (
        f"Create this week's newsletter covering the top trending "
        f"news stories of the week."
        + f"\nUse pool_max_results={collector_pool_size} as the starting "
          f"pool size when collecting candidate articles."
        + f"\nUse time_range={time_range} and search_depth={search_depth}."
        + "\nThe final newsletter must contain EXACTLY "
          f"{max_results} curated articles - not more, not fewer."
        + f"\nNewsletter Title: {newsletter_title}"
        + country_line
        + category_line
        + keywords_line
        + domains_line
        + f"\nStyle notes (do not change structure, only flavor): {style_notes}"
    )

    status_box = st.status("Running newsletter agent...", expanded=True)
    try:
        status_box.write("🔎 Collecting candidate articles...")
        status_box.write("🧠 Summarizing and scoring candidates...")
        status_box.write("🖋️ Drafting the styled HTML newsletter...")
        raw_code = main_agent(agent, user_query)
        code = raw_code.replace("```html", "").replace("```", "").strip()
        status_box.update(label="Newsletter generated!", state="complete", expanded=False)
    except Exception as e:
        status_box.update(label="Generation failed", state="error", expanded=True)
        st.error(f"Something went wrong while generating the newsletter: {e}")
        st.stop()

    plain_text = _strip_html_to_text(code)
    word_count = len(plain_text.split())
    reading_time = _reading_time_minutes(word_count)
    links = _extract_links(code)

    entry_id = str(uuid.uuid4())
    meta = {
        "country": country_choice or "Any",
        "category": category_choice or "Any",
        "language": output_language,
        "tone": tone_choice,
        "word_count": word_count,
        "reading_time": reading_time,
        "links": links,
    }
    st.session_state.last_html = code
    st.session_state.last_meta = meta
    st.session_state.history.append({
        "id": entry_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": newsletter_title,
        "html": code,
        "meta": meta,
    })

# ========== RESULTS DISPLAY (persists across reruns via session_state) =====
if st.session_state.last_html:
    code = st.session_state.last_html
    meta = st.session_state.last_meta
    plain_text = _strip_html_to_text(code)

    st.success("Newsletter generated!")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Word count", meta["word_count"])
    m2.metric("Est. reading time", f"{meta['reading_time']} min")
    m3.metric("Country", meta["country"])
    m4.metric("Category", meta["category"])

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            "⬇️ Download newsletter.html",
            data=code,
            file_name="newsletter.html",
            mime="text/html",
        )
    with dl_col2:
        st.download_button(
            "⬇️ Download as plain text",
            data=plain_text,
            file_name="newsletter.txt",
            mime="text/plain",
        )
    with dl_col3:
        with st.popover("📋 Copy HTML"):
            st.code(code, language="html")

    up_col, down_col, _ = st.columns([1, 1, 6])
    current_id = st.session_state.history[-1]["id"] if st.session_state.history else None
    if current_id:
        if up_col.button("👍", key=f"up_{current_id}"):
            st.session_state.feedback[current_id] = "up"
        if down_col.button("👎", key=f"down_{current_id}"):
            st.session_state.feedback[current_id] = "down"
        fb = st.session_state.feedback.get(current_id)
        if fb:
            st.caption(f"Thanks for the feedback ({'👍' if fb == 'up' else '👎'})!")

    st.divider()
    st.subheader("Preview")
    if preview_dark_mode:
        st.markdown(
            '<div style="background:#111;padding:12px;border-radius:8px;">',
            unsafe_allow_html=True,
        )
        st.components.v1.html(code, height=900, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.components.v1.html(code, height=900, scrolling=True)

    if meta["links"]:
        with st.expander("🔗 Sources used in this newsletter"):
            for href, text in meta["links"]:
                clean_text = re.sub(r"<[^>]+>", "", text).strip() or href
                st.markdown(f"- [{clean_text}]({href})")
