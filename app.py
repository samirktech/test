"""
AI Newsletter Generator - Streamlit App
Built using LangChain Agent (Tool Calling) + Gemini + Tavily
"""

import datetime
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

# To show web-app: complete page layout
st.set_page_config(layout="wide")

# To give title
st.title("AI NEWSLETTER GENERATOR")

st.write("""This app helps you build a curated, styled HTML newsletter
from this week's top trending news using a LangChain agent.""")

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

# ---------- 1. Country / Region ----------
REGION_PRESETS = ["World", "India", "Asia", "Europe", "United States", "Custom"]
region_choice = st.selectbox("Region / Country", REGION_PRESETS, index=0)
if region_choice == "Custom":
    region = st.text_input("Enter custom region/country", value="")
else:
    region = region_choice

# ---------- 3. Category / Topic filter ----------
CATEGORY_OPTIONS = [
    "World", "Tech", "Business", "Science", "Sports",
    "Entertainment", "Health", "Politics"
]
categories = st.multiselect(
    "Category / Topic filter",
    CATEGORY_OPTIONS,
    default=["World"],
    help="Pick one or more topics to focus the search on. Leave as World for a general mix."
)

# ---------- 4. Theme / Color scheme ----------
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


# =========== TOOL 1 ======================
def weekly_article_collector(max_results: int = 5, region: str = "World", categories: list[str] = None):
    """This function searches the web for the top trending news
    headlines published in the current week using the Tavily search
    API. Optionally focus the search on a specific region/country and
    one or more categories/topics. Returns article metadata: title,
    url, content and published date."""

    categories = categories or []
    topic_part = " and ".join(categories) if categories else "general"
    region_part = "" if not region or region.lower() == "world" else f" in {region}"

    query = f"top trending {topic_part} news headlines this week{region_part}"

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


# =========== TOOL 2 ======================
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


# =========== TOOL 3 ======================
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
      own category or relevance score.
    - At the bottom of each column, add one pale colored info box
      containing that source article's category and a bold "Read More"
      link pointing to the real article url - only add this info box if
      there is a real article category and url to put inside it, never
      as an empty filler box.
      Use the real curated article titles, summaries, categories and
      urls throughout - never placeholder/lorem ipsum text.
    - Footer: a full-width colored strip at the very bottom with small
      bold centered text reading something like "Compiled automatically
      by a multi-agent AI pipeline | Generated on {current_date}".
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


# ========== Agent Creation ================
agent = create_agent(
    model=model,
    tools=[weekly_article_collector, article_summarizer, newsletter_html_generator]
)


# ============== PROGRESS CALLBACK (2. real per-stage progress from the agent) ===============
TOOL_STAGE_LABELS = {
    "weekly_article_collector": "Collecting this week's top articles...",
    "article_summarizer": "Summarizing articles...",
    "newsletter_html_generator": "Building the HTML newsletter...",
}


class StreamlitAgentProgress(BaseCallbackHandler):
    """Listens to the agent's actual tool calls and reflects them as
    live stage updates in an st.status container, instead of a generic
    spinner."""

    def __init__(self, status):
        self.status = status
        self._summarize_count = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or ""
        if name == "article_summarizer":
            self._summarize_count += 1
            self.status.update(
                label=f"Summarizing articles... (article {self._summarize_count})",
                state="running",
            )
            self.status.write(f"Summarizing article {self._summarize_count}...")
        elif name in TOOL_STAGE_LABELS:
            self.status.update(label=TOOL_STAGE_LABELS[name], state="running")
            self.status.write(TOOL_STAGE_LABELS[name])

    def on_tool_error(self, error, **kwargs):
        self.status.write(f"A tool call hit an error: {error}")


# ============== MAIN AGENT ===============
def main_agent(agent, query, status):
    """This is the main agent, or leader agent,
    orchestrates the full newsletter workflow"""

    prompt = """Your task is to orchestrate the full newsletter workflow
    based on the instructions given below:
    1. Call the weekly_article_collector tool with max_results, region
       and categories as given, to fetch top trending news headlines
       for that region/category focus for the week.
    2. Call the article_summarizer tool separately on EACH collected
       article to get its summary, key points, category and relevance
       score.
    3. Keep the best EXACTLY 5 curated articles whenever 5 are available.
    4. Combine all the remaining curated summaries (title, summary,
       key points, category, url for each) into one collection, then
       call the newsletter_html_generator tool once with that full
       collection and the given style_hint so all curated topics
       appear in the final newsletter, styled according to style_hint.
    Give the final response output strictly in HTML, no markdowns,
    no code fences, no explanation text before or after the HTML.
    Instructions given below:
    """

    prompt = prompt + query

    handler = StreamlitAgentProgress(status)
    response = agent.invoke(
        {"messages": [{'role': 'user', 'content': prompt}]},
        config={"callbacks": [handler]},
    )
    status.update(label="Newsletter ready!", state="complete")
    code = _extract_text(response['messages'][-1])
    return code


# ========== CALLING MAIN AGENT ===============
if st.button("Generate Newsletter"):
    user_query = (
        f"Create this week's newsletter covering the top trending "
        f"news stories of the week."
        + f"\nUse max_results={max_results} when collecting articles."
        + f"\nRegion: {region if region else 'World'}"
        + f"\nCategories: {', '.join(categories) if categories else 'World (general mix)'}"
        + f"\nNewsletter Title: {newsletter_title}"
        + f"\nstyle_hint for newsletter_html_generator: {theme_style_hint}"
    )

    with st.status("Starting newsletter generation...", expanded=True) as status:
        raw_code = main_agent(agent, user_query, status)

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
