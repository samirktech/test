"""
AI Newsletter Generator - Streamlit App
Built using a LangChain multi-tool agent (Tool Calling) + Gemini + Tavily

Agent architecture:
    Leader agent (create_agent) orchestrates 3 tools it can call in any
    order / any number of times it decides is needed:
        1. weekly_article_collector  -> Tavily web search
        2. article_summarizer        -> Gemini summarization
        3. newsletter_html_generator -> Gemini HTML generation
"""

import json
import time
import datetime

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

APP_VERSION = "2.0"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Newsletter Generator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL STYLING
# =========================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --ink: #0f172a;
            --ink-soft: #475569;
            --muted: #94a3b8;
            --border: #e2e8f0;
            --accent: #2952e3;
            --accent-dark: #1e3fc4;
            --accent-tint: #eef1fd;
            --surface: #ffffff;
            --canvas: #f7f8fb;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            color: var(--ink);
        }

        /* ---- overall page ---- */
        .stApp {
            background: var(--canvas);
        }
        .block-container {
            padding-top: 2rem;
        }

        /* ---- masthead ---- */
        .masthead {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem 1.75rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 10px;
            margin-bottom: 1.4rem;
        }
        .masthead-left { display: flex; align-items: center; gap: 0.9rem; }
        .masthead-icon {
            width: 46px; height: 46px;
            border-radius: 9px;
            background: var(--accent);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.35rem;
            flex-shrink: 0;
        }
        .masthead h1 {
            font-size: 1.32rem;
            font-weight: 700;
            margin: 0;
            color: var(--ink);
            letter-spacing: -0.01px;
        }
        .masthead p {
            font-size: 0.86rem;
            color: var(--ink-soft);
            margin: 0.15rem 0 0 0;
            max-width: 560px;
            line-height: 1.4;
        }
        .masthead .version-tag {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--accent);
            background: var(--accent-tint);
            border: 1px solid #d6ddf9;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            white-space: nowrap;
        }

        .capability-row { margin-bottom: 1.6rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
        .capability-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--ink-soft);
            padding: 0.35rem 0.75rem;
            border-radius: 7px;
            font-size: 0.78rem;
            font-weight: 500;
        }

        /* ---- section card ---- */
        .section-card {
            background: var(--surface);
            border-radius: 10px;
            padding: 1.3rem 1.5rem;
            border: 1px solid var(--border);
            margin-bottom: 1.1rem;
        }
        .section-title {
            font-weight: 700;
            font-size: 0.92rem;
            color: var(--ink);
            margin-bottom: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .section-title .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px; height: 20px;
            background: var(--accent);
            color: #fff;
            border-radius: 5px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: none;
            letter-spacing: 0;
        }

        /* ---- history card ---- */
        .history-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
        }
        .history-title {
            font-weight: 700;
            color: var(--ink);
            font-size: 0.95rem;
        }
        .history-meta {
            color: var(--muted);
            font-size: 0.76rem;
            margin-top: 0.2rem;
        }
        .tag-chip {
            display: inline-block;
            background: var(--accent-tint);
            color: var(--accent);
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.1rem 0.5rem;
            border-radius: 5px;
            margin-right: 0.3rem;
            border: 1px solid #d6ddf9;
        }

        /* ---- sidebar ---- */
        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid #1c2740;
        }
        section[data-testid="stSidebar"] * {
            color: #cbd5e1 !important;
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #f1f5f9 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] input {
            color: #0f172a !important;
        }
        section[data-testid="stSidebar"] .stAlert {
            border-radius: 7px;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #1c2740;
        }

        /* ---- buttons ---- */
        .stButton > button {
            background: var(--accent);
            color: white;
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 0.6rem 1.3rem;
            font-weight: 600;
            font-size: 0.92rem;
            transition: background 0.12s ease-in-out;
            box-shadow: none;
        }
        .stButton > button:hover {
            background: var(--accent-dark);
            border-color: var(--accent-dark);
        }
        .stButton > button:disabled {
            background: #e2e8f0;
            border-color: #e2e8f0;
            color: #94a3b8;
        }

        .stDownloadButton > button {
            background: var(--surface);
            color: var(--ink);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-weight: 600;
        }
        .stDownloadButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* ---- metric chips ---- */
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: var(--muted) !important;
        }

        /* ---- tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--ink-soft);
            padding: 0.6rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
        }

        /* ---- copy button (custom html component) ---- */
        .copy-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--ink);
            font-weight: 600;
            padding: 0.55rem 1.1rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            width: 100%;
        }
        .copy-btn:hover { border-color: var(--accent); color: var(--accent); }

        /* ---- misc ---- */
        hr { border-color: var(--border); }
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MASTHEAD
# =========================================================
st.markdown(
    f"""
    <div class="masthead">
        <div class="masthead-left">
            <div class="masthead-icon">📰</div>
            <div>
                <h1>AI Newsletter Generator</h1>
                <p>A LangChain multi-tool agent researches this week's trending stories,
                scores and summarizes them with Gemini, then designs a styled HTML newsletter.</p>
            </div>
        </div>
        <div class="version-tag">v{APP_VERSION}</div>
    </div>
    <div class="capability-row">
        <span class="capability-chip">🔎 Tavily Search</span>
        <span class="capability-chip">✨ Gemini</span>
        <span class="capability-chip">🧠 LangChain Tool-Calling Agent</span>
        <span class="capability-chip">🎨 Auto-styled HTML Output</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================
if "newsletter_html" not in st.session_state:
    st.session_state.newsletter_html = None
if "generation_seconds" not in st.session_state:
    st.session_state.generation_seconds = None
if "generated_title" not in st.session_state:
    st.session_state.generated_title = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, most recent first

# =========================================================
# SIDEBAR - API KEYS
# =========================================================
st.sidebar.markdown("## 🔑 API Keys")
st.sidebar.caption("Your keys are only used for this session and are never stored.")

TAVILY_API_KEY = st.sidebar.text_input(
    "Tavily API Key", type="password", placeholder="tvly-..."
)
GOOGLE_API_KEY = st.sidebar.text_input(
    "Gemini API Key", type="password", placeholder="AIza..."
)

with st.sidebar.expander("Where do I get these keys?"):
    st.markdown(
        "- **Tavily** → [app.tavily.com](https://app.tavily.com)\n"
        "- **Gemini** → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
    )

all_API = [TAVILY_API_KEY, GOOGLE_API_KEY]
keys_ready = all(all_API)

if keys_ready:
    st.sidebar.success("✅ API keys loaded")
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
    )
else:
    st.sidebar.warning("⏳ Enter both API keys to continue")

st.sidebar.divider()
st.sidebar.markdown("## 🎛️ Writing Style")
tone_choice = st.sidebar.selectbox(
    "Tone",
    ["Balanced & Professional", "Formal / Corporate", "Casual & Punchy", "Concise & Minimal"],
    index=0,
    help="Shapes the summarizer's voice and the newsletter's tagline/heading style.",
)

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ How it works")
st.sidebar.markdown(
    "1. **Collector tool** pulls a pool of trending headlines\n"
    "2. **Summarizer tool** scores & condenses each one\n"
    "3. Agent keeps the **best 5** by relevance\n"
    "4. **HTML tool** designs the final newsletter page"
)

max_results = 5
collector_pool_size = 10  # fetch a larger pool so exactly 5 curated articles are always available

# =========================================================
# NAV TABS
# =========================================================
tab_generate, tab_history, tab_about = st.tabs(
    ["✨ Generate", f"🕓 History ({len(st.session_state.history)})", "ℹ️ About"]
)

# =========================================================
# TOOL 1
# =========================================================
def weekly_article_collector(max_results=5, country=None, category=None):
    """This function searches the web for the top trending news
    headlines published in the current week using the Tavily search
    API. Optionally restricts results to a specific country and/or
    category. Returns article metadata: title, url, content and
    published date."""

    query_parts = ["top trending"]
    if category:
        query_parts.append(f"{category}")
    query_parts.append("news headlines")
    if country:
        query_parts.append(f"in {country}")
    query_parts.append("this week")
    query = " ".join(query_parts)

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=query,
        topic="news",
        time_range="week",
        max_results=max_results,
        include_answer=False,
    )

    articles = []
    for result in response.get("results", []):
        articles.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "published_date": result.get("published_date", "N/A"),
        })
    return articles


# =========================================================
# TOOL 2
# =========================================================
def article_summarizer(article_text, article_title="Untitled", tone="Balanced & Professional"):
    """This function takes article text or url content and
    produces a concise summary, key points, category
    and relevance score (out of 10) using LLM,
    given article title, content and desired tone."""

    prompt = f"""You are a professional newsletter editor writing in a
    "{tone}" voice. Summarize the article below in 3-4 concise lines
    matching that tone, then list 2-3 key points as bullets, assign a
    single category (Tech/Business/Science/World/Other) and give
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


# =========================================================
# TOOL 3
# =========================================================
def newsletter_html_generator(curated_summaries, newsletter_title="Weekly Newsletter", tone="Balanced & Professional"):
    """This function converts curated article summaries
    into a styled html newsletter template suitable
    for email or web publishing, given curated summaries
    text, newsletter title and desired tone."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")

    prompt = f"""Convert the curated article summaries below into a single
    self-contained HTML page styled like a printed magazine/school
    newsletter front page. Write all headings, taglines and copy in a
    "{tone}" voice. Return a full HTML document with a <style>
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


# =========================================================
# AGENT CREATION
# =========================================================
def build_agent():
    return create_agent(
        model=model,
        tools=[weekly_article_collector, article_summarizer, newsletter_html_generator],
    )


def main_agent(agent, query):
    """This is the main agent, or leader agent,
    orchestrates the full newsletter workflow"""

    prompt = """Your task is to orchestrate the full newsletter workflow
    based on the instructions given below:
    1. Call the weekly_article_collector tool with the pool_max_results
       given, and with the country and category as given (if any), to
       fetch a pool of candidate top trending news headlines for the
       week. If a country is given, ONLY include news from that
       country. If a category is given, ONLY include news from that
       category.
    2. The final newsletter MUST contain EXACTLY 5 curated articles,
       never more and never fewer. If the pool returned fewer than 5
       articles, call weekly_article_collector again with a larger
       pool_max_results (and, if needed, a broader/looser version of
       the country or category constraint) until at least 5 usable
       articles are available.
    3. Call the article_summarizer tool separately on EACH candidate
       article (passing the requested tone) to get its summary, key
       points, category and relevance score.
    4. From the summarized candidates, keep EXACTLY the best 5
       curated articles by relevance score.
    5. Combine those EXACTLY 5 curated summaries (title, summary,
       key points, category, url for each) into one collection, then
       call the newsletter_html_generator tool once with that full
       collection (passing the requested tone) so all 5 curated
       articles appear in the final newsletter.
    Give the final response output strictly in HTML, no markdowns,
    no code fences, no explanation text before or after the HTML.
    Instructions given below:
    """

    prompt = prompt + query

    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    code = _extract_text(response["messages"][-1])
    return code


def estimate_read_time(html):
    """Rough reading-time estimate based on word count of the raw HTML text."""
    word_count = len(html.split())
    minutes = max(1, round(word_count / 220))
    return minutes


def render_copy_button(html_code, key_suffix=""):
    payload = json.dumps(html_code)
    component_html = f"""
    <button class="copy-btn" onclick="copyCode{key_suffix}()">📋 Copy HTML to clipboard</button>
    <script>
        function copyCode{key_suffix}() {{
            const text = {payload};
            navigator.clipboard.writeText(text);
        }}
    </script>
    """
    st.components.v1.html(component_html, height=48)


# =========================================================
# TAB: GENERATE
# =========================================================
with tab_generate:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Newsletter Details</div>', unsafe_allow_html=True)

    newsletter_title = st.text_input(
        "Newsletter Title", value="Weekly Digest", placeholder="e.g. The Monday Brief"
    )

    col1, col2 = st.columns(2)

    with col1:
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
            "Any",
            "Custom",
        ]
        selected_country = st.selectbox("🌍 Country", country_options, index=0)

        if selected_country == "Custom":
            country_choice = st.text_input("Enter Country", value="")
        elif selected_country == "Any":
            country_choice = None
        else:
            country_choice = selected_country

    with col2:
        category_options = ["Tech", "Business", "Science", "World", "Any"]
        selected_category = st.selectbox("🏷️ Category", category_options, index=0)
        category_choice = None if selected_category == "Any" else selected_category

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 Generate</div>', unsafe_allow_html=True)

    gen_col, clear_col = st.columns([1, 1])
    generate_clicked = gen_col.button(
        "✨ Generate Newsletter", disabled=not keys_ready, use_container_width=True
    )
    clear_clicked = clear_col.button(
        "🗑️ Clear Result", use_container_width=True, disabled=st.session_state.newsletter_html is None
    )

    if not keys_ready:
        st.caption("Enter your Tavily and Gemini API keys in the sidebar to enable generation.")

    st.markdown("</div>", unsafe_allow_html=True)

    if clear_clicked:
        st.session_state.newsletter_html = None
        st.session_state.generation_seconds = None
        st.session_state.generated_title = None
        st.rerun()

    if generate_clicked:
        if not newsletter_title.strip():
            st.error("Please give your newsletter a title before generating.")
        elif selected_country == "Custom" and not country_choice.strip():
            st.error("You selected Custom country — please type a country name.")
        else:
            status_box = st.status("Running the newsletter agent...", expanded=True)
            start_time = time.time()
            try:
                status_box.write("🔎 Collecting this week's trending headlines...")
                status_box.write("🧮 Summarizing and scoring each candidate article...")
                status_box.write("🎨 Designing the final HTML newsletter layout...")

                agent = build_agent()

                if country_choice:
                    country_line = f"\nCountry: {country_choice} (ONLY use news from this country)."
                else:
                    country_line = "\nCountry: Any (do not restrict to a single country)."

                if category_choice:
                    category_line = f"\nCategory: {category_choice} (ONLY use news from this category)."
                else:
                    category_line = "\nCategory: Any (do not restrict to a single category)."

                user_query = (
                    f"Create this week's newsletter covering the top trending "
                    f"news stories of the week."
                    + f"\nUse pool_max_results={collector_pool_size} as the starting "
                      f"pool size when collecting candidate articles."
                    + "\nThe final newsletter must contain EXACTLY "
                      f"{max_results} curated articles - not more, not fewer."
                    + f"\nNewsletter Title: {newsletter_title}"
                    + f"\nTone: {tone_choice} (use this voice for summaries and newsletter copy)."
                    + country_line
                    + category_line
                )

                raw_code = main_agent(agent, user_query)
                code = raw_code.replace("```html", "").replace("```", "").strip()

                elapsed = round(time.time() - start_time, 1)
                st.session_state.newsletter_html = code
                st.session_state.generation_seconds = elapsed
                st.session_state.generated_title = newsletter_title

                st.session_state.history.insert(0, {
                    "title": newsletter_title,
                    "html": code,
                    "timestamp": datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    "seconds": elapsed,
                    "country": selected_country,
                    "category": selected_category,
                    "tone": tone_choice,
                })
                st.session_state.history = st.session_state.history[:10]

                status_box.update(label="Newsletter ready!", state="complete", expanded=False)

            except Exception as exc:
                status_box.update(label="Generation failed", state="error", expanded=True)
                st.error(
                    "Something went wrong while generating the newsletter. "
                    "Double-check your API keys and try again.\n\n"
                    f"**Details:** {exc}"
                )

    # ---- Results ----
    if st.session_state.newsletter_html:
        code = st.session_state.newsletter_html

        st.success(f"🎉 \"{st.session_state.generated_title}\" is ready!")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles curated", "5")
        m2.metric("Generation time", f"{st.session_state.generation_seconds}s")
        m3.metric("Output size", f"{len(code) // 1024} KB")
        m4.metric("Est. read time", f"{estimate_read_time(code)} min")

        dl_col, copy_col = st.columns([1, 1])
        with dl_col:
            st.download_button(
                "⬇️ Download newsletter.html",
                data=code,
                file_name=f"{st.session_state.generated_title.strip().replace(' ', '_') or 'newsletter'}.html",
                mime="text/html",
                use_container_width=True,
            )
        with copy_col:
            render_copy_button(code, key_suffix="main")

        st.divider()

        tab_preview, tab_source = st.tabs(["👀 Preview", "🧾 HTML Source"])
        with tab_preview:
            st.components.v1.html(code, height=900, scrolling=True)
        with tab_source:
            st.code(code, language="html")

# =========================================================
# TAB: HISTORY
# =========================================================
with tab_history:
    if not st.session_state.history:
        st.info("No newsletters generated yet this session. Generate one from the ✨ Generate tab to see it here.")
    else:
        st.caption(f"Last {len(st.session_state.history)} newsletter(s) generated this session (not saved after you close the tab).")
        for i, item in enumerate(st.session_state.history):
            st.markdown(
                f"""
                <div class="history-card">
                    <div class="history-title">📰 {item['title']}</div>
                    <div class="history-meta">{item['timestamp']} · {item['seconds']}s ·
                        <span class="tag-chip">{item['country']}</span>
                        <span class="tag-chip">{item['category']}</span>
                        <span class="tag-chip">{item['tone']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            hcol1, hcol2, hcol3 = st.columns([1, 1, 2])
            with hcol1:
                st.download_button(
                    "⬇️ Download",
                    data=item["html"],
                    file_name=f"{item['title'].strip().replace(' ', '_')}.html",
                    mime="text/html",
                    key=f"hist_dl_{i}",
                    use_container_width=True,
                )
            with hcol2:
                if st.button("👀 Load into Preview", key=f"hist_load_{i}", use_container_width=True):
                    st.session_state.newsletter_html = item["html"]
                    st.session_state.generation_seconds = item["seconds"]
                    st.session_state.generated_title = item["title"]
                    st.rerun()
            with hcol3:
                with st.expander("Preview inline"):
                    st.components.v1.html(item["html"], height=500, scrolling=True)

# =========================================================
# TAB: ABOUT
# =========================================================
with tab_about:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Agent Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
This app is powered by a **multi-tool LangChain agent**, not a fixed linear
pipeline. A single leader agent (`create_agent`) decides, at runtime, how many
times to call each tool and in what order — for example, re-running the
collector with a wider pool if fewer than 5 usable articles come back.

**Tools available to the agent:**
| Tool | Purpose |
|---|---|
| `weekly_article_collector` | Searches Tavily for this week's trending headlines, filtered by country/category |
| `article_summarizer` | Uses Gemini to summarize, categorize and score each candidate article |
| `newsletter_html_generator` | Uses Gemini to lay out the final 5 curated articles into a styled HTML page |

**Stack:** Streamlit · LangChain (`create_agent` / tool calling) · Google Gemini · Tavily Search API
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔒 Privacy</div>', unsafe_allow_html=True)
    st.markdown(
        "API keys are held only in your browser session's memory for this run "
        "and are sent directly to Google/Tavily — never logged or stored by this app."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="text-align:center; color:#8b8ba7; font-size:0.82rem; margin-top:2rem;">
        Built with a LangChain multi-tool agent · Tavily Search · Google Gemini · v{APP_VERSION}
    </div>
    """,
    unsafe_allow_html=True,
)
