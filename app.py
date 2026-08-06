"""
AI Newsletter Generator - Streamlit App
Built using Multi-Agent Orchestration (three specialized, single-tool
LangChain agents coordinated by a supervisor function) + Gemini + Tavily.

Architecture:
    Collector Agent   -> owns weekly_article_collector tool only
    Summarizer Agent  -> owns article_summarizer tool only
    Editor Agent      -> owns newsletter_html_generator tool only
    orchestrate_newsletter() is the supervisor that calls each agent in
    turn and passes structured data between them. No single agent has
    access to more than one tool, and no agent decides the overall
    workflow - the supervisor does.
"""

import ast
import json
import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

# To show web-app: complete page layout
st.set_page_config(layout="wide")

# To give title
st.title("AI NEWSLETTER GENERATOR")

st.write("""This app helps you build a curated, styled HTML newsletter
from this week's top trending news for one country and one category,
using a multi-agent pipeline (Collector -> Summarizer -> Editor).""")

st.sidebar.title("Fill Important Details")

# ============ API KEYS ===================
TAVILY_API_KEY = st.sidebar.text_input("Tavily-API", type="password")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini-API", type="password")

all_API = [TAVILY_API_KEY, GOOGLE_API_KEY]

if not all(all_API):
    st.error("Must give API keys")
    st.stop()
elif all(all_API):
    st.success("API KEYS LOADED SUCCESSFULLY")
    # =========== MODEL CREATION ==============
    model = ChatGoogleGenerativeAI(
        model='gemini-3.5-flash-lite',
        google_api_key=GOOGLE_API_KEY
    )
else:
    st.info("PASS ALL API-KEYS")

max_results = 5

# ==================GET USER INFO=====================
st.markdown("### NEWSLETTER DETAILS")
newsletter_title = st.text_input("Newsletter Title", value="Weekly Digest")

# ---------- 1. Country (no region/continent options, single country only) ----------
COUNTRY_PRESETS = [
    "India", "United States", "United Kingdom", "Canada", "Australia",
    "Germany", "France", "Japan", "Singapore", "United Arab Emirates",
    "Custom",
]
country_choice = st.selectbox("Country", COUNTRY_PRESETS, index=0)
if country_choice == "Custom":
    country = st.text_input("Enter custom country", value="")
else:
    country = country_choice

if not country.strip():
    st.warning("Please enter a country to continue.")
    st.stop()

# ---------- 2. Category / Topic filter (single choice only) ----------
CATEGORY_OPTIONS = [
    "World", "Tech", "Business", "Science", "Sports",
    "Entertainment", "Health", "Politics"
]
category = st.selectbox(
    "Category / Topic",
    CATEGORY_OPTIONS,
    index=0,
    help="Pick exactly one topic. The newsletter will contain news from "
         "this category only.",
)

# ---------- 3. Theme / Color scheme ----------
THEME_OPTIONS = {
    "Classic Newspaper": (
        "Classic black-and-white broadsheet newspaper look: cream/off-white "
        "background, black serif headings, thin black rule lines, minimal color "
        "used only for one or two accent highlights."
    ),
    "Dark Mode": (
        "Dark mode aesthetic: near-black background (#111 range), light "
        "off-white body text, vibrant accent color (electric blue or amber) "
        "for headings, banners and highlight lines, subtle lighter-gray borders."
    ),
    "Pastel": (
        "Soft pastel aesthetic: light pastel background (blush pink or mint), "
        "rounded section boxes, pastel accent bands (lavender, peach, baby blue), "
        "dark charcoal text for strong contrast, friendly rounded sans-serif feel."
    ),
    "Corporate": (
        "Clean corporate/business aesthetic: white background, navy blue and "
        "slate gray accents, structured grid with clear dividing lines, "
        "professional sans-serif headings, minimal decoration."
    ),
}
theme_choice = st.selectbox("Newsletter Theme", list(THEME_OPTIONS.keys()), index=0)
theme_style_hint = THEME_OPTIONS[theme_choice]


# =========== TOOL 1 (owned only by the Collector Agent) ======================
def weekly_article_collector(max_results: int = 5, country: str = "", category: str = "World"):
    """This function searches the web for the top trending news
    headlines published in the current week using the Tavily search
    API, restricted to a single country and a single category/topic.
    Returns article metadata: title, url, content and published date."""

    query = f"top trending {category} news headlines this week in {country}"

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


# =========== TOOL 2 (owned only by the Summarizer Agent) ======================
def article_summarizer(article_text, article_title="Untitled"):
    """This function takes article text or url content and
    produces a concise summary, key points, category
    and relevance score (out of 10) using LLM,
    given article title and content"""

    prompt = f"""You are a professional newsletter editor.
    Summarize the article below in 3-4 concise lines,
    then list 2-3 key points as bullets, assign a single
    category (Tech/Business/Science/World/Other) and give
    a relevance score out of 10 for a general audience.

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


# =========== TOOL 3 (owned only by the Editor Agent) ======================
def newsletter_html_generator(curated_summaries, newsletter_title="Weekly Newsletter", style_hint=""):
    """This function converts curated article summaries
    into a styled html newsletter template suitable
    for email or web publishing, given curated summaries
    text, newsletter title, and an optional style/theme hint
    describing the desired color scheme and visual mood."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")

    prompt = f"""Convert the curated article summaries below into a single
    self-contained HTML page styled like a printed magazine/school
    newsletter front page. Return a full HTML document with a <style>
    block in the <head> (CSS does not need to be inline), max content
    width around 900px, centered on the page with a bordered page frame.

    THEME / STYLE DIRECTION (follow this closely for colors and feel):
    {style_hint}

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
      stating that this covers this week's top stories, naming the
      single country and single category the stories come from (do not
      imply multiple countries or multiple categories are covered).
    - Below the banner, use a CSS grid with exactly 2 columns
      (display: grid; grid-template-columns: 1fr 1fr; column-gap and
      row-gap around 24px) to lay out one section per curated article.
      Do NOT create a single "top story" block - turn every curated
      article into its own section: a bold uppercase heading (short,
      based on the article's own title) followed by a paragraph using
      that article's summary and key points as body text. All articles
      belong to the same single category, so do not invent separate
      category labels that contradict that.
    - CRITICAL - no empty or blank cells: count the curated articles
      first. If the count is odd, make the LAST article's section span
      both columns (grid-column: 1 / -1) instead of leaving an empty
      cell next to it. Never render an empty box, an empty <div>, or a
      solid-colored block with no text in it anywhere on the page -
      every colored box must contain real content from the curated
      summaries. If there are not enough curated summaries to fill a
      section you were going to add, simply omit that section instead
      of leaving it blank.
    - For 2-3 of the sections, add a short colored accent line
      (in a highlight color) above the paragraph showing that article's
      relevance score.
    - At the bottom of each column, add one pale colored info box
      containing a bold "Read More" link pointing to the real article
      url - only add this info box if there is a real article url to
      put inside it, never as an empty filler box.
      Use the real curated article titles, summaries and urls
      throughout - never placeholder/lorem ipsum text.
    - Footer: a full-width colored strip at the very bottom with small
      bold centered text reading something like "Compiled automatically
      by a multi-agent AI pipeline | Generated on {current_date}".
    Give final response strictly in HTML only, no markdown, no code fences.

    Newsletter Title: {newsletter_title}
    Generated On: {current_date}
    Curated Summaries (single country, single category, this week): {curated_summaries}
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


def _get_tool_messages(messages, tool_name):
    """Pull out the raw outputs a specific tool produced inside one
    agent's message history. Works whether messages are LangChain
    message objects or plain dicts, since agents in this app are each
    restricted to exactly one tool and we trust that tool's structured
    output over the agent's own restated text."""
    outputs = []
    for m in messages:
        if isinstance(m, dict):
            m_type = m.get("type") or m.get("role")
            m_name = m.get("name")
            m_content = m.get("content")
        else:
            m_type = getattr(m, "type", None)
            m_name = getattr(m, "name", None)
            m_content = getattr(m, "content", None)
        if m_type == "tool" and m_name == tool_name:
            outputs.append(m_content)
    return outputs


def _parse_tool_output(raw):
    """Best-effort parse of a tool's raw output back into a Python
    object (list/dict), falling back to the original value."""
    if isinstance(raw, (list, dict)):
        return raw
    if not isinstance(raw, str):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw


# ========== Multi-Agent Setup ================
# Each agent is scoped to exactly one tool and one job. There is no
# single "do everything" agent - the three agents below cannot see or
# call each other's tools. Coordination across them is handled entirely
# by the orchestrate_newsletter() supervisor function further down.
collector_agent = create_agent(model=model, tools=[weekly_article_collector])
summarizer_agent = create_agent(model=model, tools=[article_summarizer])
editor_agent = create_agent(model=model, tools=[newsletter_html_generator])


# ============== AGENT 1: COLLECTOR ===============
def run_collector_agent(country, category, max_results, status):
    status.update(label="Agent 1/3 - Collector: gathering this week's headlines...", state="running")
    status.write(f"Collector Agent searching '{category}' news in {country}...")

    instruction = (
        f"Call the weekly_article_collector tool exactly once with "
        f"max_results={max_results}, country='{country}' and "
        f"category='{category}' to fetch this week's top trending "
        f"headlines for that single country and single category."
    )
    result = collector_agent.invoke({"messages": [{"role": "user", "content": instruction}]})

    tool_msgs = _get_tool_messages(result["messages"], "weekly_article_collector")
    articles = _parse_tool_output(tool_msgs[-1]) if tool_msgs else []
    if not isinstance(articles, list):
        articles = []

    status.write(f"Collector Agent found {len(articles)} article(s).")
    return articles


# ============== AGENT 2: SUMMARIZER ===============
def run_summarizer_agent(articles, status):
    status.update(label="Agent 2/3 - Summarizer: summarizing each article...", state="running")
    summaries = []

    for i, article in enumerate(articles, start=1):
        title = article.get("title", "Untitled")
        status.write(f"Summarizer Agent processing article {i}/{len(articles)}: {title}")

        instruction = (
            "Call the article_summarizer tool exactly once, passing the "
            "article content below as article_text and the title as "
            "article_title.\n\n"
            f"Article Title: {title}\n"
            f"Article Content: {article.get('content', '')}"
        )
        result = summarizer_agent.invoke({"messages": [{"role": "user", "content": instruction}]})

        tool_msgs = _get_tool_messages(result["messages"], "article_summarizer")
        summary_text = tool_msgs[-1] if tool_msgs else _extract_text(result["messages"][-1])

        summaries.append({
            "title": title,
            "url": article.get("url", ""),
            "summary": summary_text,
        })

    return summaries


# ============== AGENT 3: EDITOR ===============
def run_editor_agent(summaries, newsletter_title, style_hint, status):
    status.update(label="Agent 3/3 - Editor: building the HTML newsletter...", state="running")
    status.write("Editor Agent assembling the final newsletter...")

    curated_blob = "\n\n".join(
        f"Title: {s['title']}\nURL: {s['url']}\nSummary: {s['summary']}"
        for s in summaries
    )
    instruction = (
        "Call the newsletter_html_generator tool exactly once, passing "
        f"the curated_summaries below, newsletter_title='{newsletter_title}' "
        "and the given style_hint.\n\n"
        f"style_hint: {style_hint}\n\ncurated_summaries:\n{curated_blob}"
    )
    result = editor_agent.invoke({"messages": [{"role": "user", "content": instruction}]})

    tool_msgs = _get_tool_messages(result["messages"], "newsletter_html_generator")
    if tool_msgs:
        return tool_msgs[-1]
    return _extract_text(result["messages"][-1])


# ============== SUPERVISOR: ORCHESTRATES THE 3 AGENTS ===============
def orchestrate_newsletter(country, category, max_results, newsletter_title, style_hint, status):
    """Supervisor function. Runs the Collector, Summarizer and Editor
    agents in sequence, passing structured data between them. This is
    the coordination layer of the multi-agent system - none of the
    three agents knows about the others or about the overall workflow;
    only the supervisor does."""

    articles = run_collector_agent(country, category, max_results, status)
    if not articles:
        status.update(label="No articles found", state="error")
        return (
            "<html><body style='color:#111;background:#fff;'>"
            "<p>No articles were found for this country/category this week. "
            "Try a different country or category.</p></body></html>"
        )

    curated_articles = articles[:5]
    summaries = run_summarizer_agent(curated_articles, status)
    html_code = run_editor_agent(summaries, newsletter_title, style_hint, status)

    status.update(label="Newsletter ready!", state="complete")
    return html_code


# ========== CALLING THE MULTI-AGENT PIPELINE ===============
if st.button("Generate Newsletter"):
    with st.status("Starting multi-agent newsletter generation...", expanded=True) as status:
        raw_code = orchestrate_newsletter(
            country=country,
            category=category,
            max_results=max_results,
            newsletter_title=newsletter_title,
            style_hint=theme_style_hint,
            status=status,
        )

    code = raw_code.replace("```html", "").replace("```", "").strip()

    st.success("Newsletter generated!")

    st.download_button(
        "Download newsletter.html",
        data=code,
        file_name="newsletter.html",
        mime="text/html",
    )

    st.divider()
    st.subheader("Preview")
    st.components.v1.html(code, height=900, scrolling=True)
