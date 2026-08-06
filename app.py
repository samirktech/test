"""
AI Newsletter Generator - Streamlit App
Built using LangChain (Tool Calling) + Gemini + Tavily
"""

import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# To show web-app: complete page layout
st.set_page_config(layout="wide")

# To give title
st.title("AI NEWSLETTER GENERATOR")

st.write("""This app helps you build a curated, styled HTML newsletter
from this week's top trending news using LangChain tool-calling.""")

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

# ==================GET USER INFO=====================
st.markdown("### NEWSLETTER DETAILS")

newsletter_title = st.text_input("Newsletter Title", value="Weekly Digest")

# ---------- Country / Region ----------
COUNTRY_PRESETS = [
    "World",
    "India",
    "USA",
    "UK",
    "Asia",
    "Europe",
    "Middle East",
    "Africa",
    "Southeast Asia",
    "Latin America",
    "Custom (type below)",
]

col_c1, col_c2 = st.columns([1, 1])

with col_c1:
    country_choice = st.selectbox(
        "Country / Region Focus",
        options=COUNTRY_PRESETS,
        index=0,
        help='Pick "World" for global coverage, a preset region/country, '
             'or "Custom (type below)" to type your own.'
    )

with col_c2:
    custom_country = st.text_input(
        "Custom Country / Region (only used if 'Custom' is selected above)",
        value="",
        placeholder='e.g. "Japan", "Middle East", "Scandinavia"'
    )

if country_choice == "Custom (type below)":
    selected_country = (custom_country or "").strip() or "World"
else:
    selected_country = country_choice

# ---------- Category / Topic ----------
CATEGORY_OPTIONS = [
    "Any / Mixed",
    "Technology",
    "Business",
    "Science",
    "Sports",
    "Entertainment",
    "Health",
    "Politics",
]

selected_category = st.selectbox(
    "Category / Topic Filter",
    options=CATEGORY_OPTIONS,
    index=0,
    help='Pick a specific topic to focus the newsletter on, or leave as '
         '"Any / Mixed" for a general mix of top stories.'
)

# ---------- Number of Articles ----------
max_results = st.slider(
    "Number of Articles",
    min_value=3,
    max_value=8,
    value=5,
    help="Controls how many articles are fetched, summarized, and "
         "included in the final newsletter. The layout automatically "
         "adapts (including odd counts) so there are never blank boxes."
)

# ---------- Theme ----------
THEME_OPTIONS = [
    "Classic Newspaper (Light)",
    "Dark Mode",
    "Pastel",
    "Corporate / Professional",
]

selected_theme = st.selectbox(
    "Newsletter Theme / Color Scheme",
    options=THEME_OPTIONS,
    index=0,
    help="Controls the color palette and visual style used in the "
         "generated HTML newsletter."
)

THEME_STYLE_HINTS = {
    "Classic Newspaper (Light)": (
        "Off-white/cream page background (around #f5f2ea), black or "
        "very dark gray headline and body text, a deep maroon or navy "
        "accent color for the sub-banner, accent lines and footer. "
        "Classic newspaper nameplate feel."
    ),
    "Dark Mode": (
        "Dark background for the whole page (around #12141c or #1a1a1a), "
        "light gray/off-white text (around #e8e8e8) for headlines and "
        "body copy, a vivid accent color (electric blue, amber, or teal) "
        "for the sub-banner, accent lines and footer. The pale info boxes "
        "at the bottom of each column should still use light backgrounds "
        "with dark text so they stay clearly readable against the dark page."
    ),
    "Pastel": (
        "Soft pastel page background (light lavender, mint, or blush, "
        "around #f5f0fa), dark charcoal text for readability, pastel "
        "pink/blue/yellow accent bands and rounded corners on boxes for "
        "a soft, friendly look."
    ),
    "Corporate / Professional": (
        "Clean white or very light gray page background, navy or "
        "steel-blue accent bands, dark slate-gray text, minimal rounded "
        "corners, understated sans-serif styling for a professional, "
        "business-report look."
    ),
}


# =========== TOOL 1 ======================
def weekly_article_collector(max_results=5, country="World", category="Any / Mixed"):
    """This function searches the web for the top trending news
    headlines published in the current week using the Tavily search
    API. If country is "World" (or empty/not specified), it fetches
    general top trending news from all over the world. If a specific
    country or region is given (e.g. "India", "Middle East"), it
    fetches top trending news headlines only from/about that
    country/region. If category is not "Any / Mixed", results are
    further focused on that topic (e.g. Technology, Sports). Returns
    article metadata: title, url, content and published date."""

    country = (country or "World").strip()
    category = (category or "Any / Mixed").strip()

    query_parts = ["top trending"]
    if category.lower() not in ("", "any / mixed", "any", "mixed"):
        query_parts.append(category.lower())
    query_parts.append("news headlines this week")

    if country.lower() not in ("", "world", "global", "all", "any"):
        query_parts.append(f"in {country}")

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


# =========== TOOL 2 ======================
def article_summarizer(article_text, article_title="Untitled"):
    """This function takes article text or url content and
    produces a concise summary, key points, category
    and relevance score (out of 10) using LLM,
    given article title and content"""

    prompt = f"""You are a professional newsletter editor.
    Summarize the article below in 3-4 concise lines,
    then list 2-3 key points as bullets, assign a single
    category (Tech/Business/Science/World/Sports/Entertainment/Health/
    Politics/Other) and give a relevance score out of 10 for a general
    audience.

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
def newsletter_html_generator(curated_summaries, newsletter_title="Weekly Newsletter",
                               theme="Classic Newspaper (Light)", article_count=None):
    """This function converts curated article summaries
    into a styled html newsletter template suitable
    for email or web publishing, given curated summaries
    text, newsletter title, a color/theme style, and the exact
    number of curated articles (so the grid layout matches reality
    and never leaves blank cells)."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")
    style_hint = THEME_STYLE_HINTS.get(theme, THEME_STYLE_HINTS["Classic Newspaper (Light)"])
    if article_count is None:
        article_count = curated_summaries.count("Article ")

    prompt = f"""Convert the curated article summaries below into a single
    self-contained HTML page styled like a printed magazine/school
    newsletter front page. Return a full HTML document with a <style>
    block in the <head> (CSS does not need to be inline), max content
    width around 900px, centered on the page with a bordered page frame.

    COLOR / THEME DIRECTION FOR THIS NEWSLETTER:
    {style_hint}

    CRITICAL - TEXT MUST ALWAYS BE VISIBLE (applies no matter which
    theme colors are used above):
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
      summarizing that this covers this week's top stories (infer a
      short tagline from the actual mix of categories present in the
      curated summaries below - do not hardcode a single subject unless
      every curated article genuinely belongs to that one subject).
    - Below the banner, use a CSS grid with exactly 2 columns
      (display: grid; grid-template-columns: 1fr 1fr; column-gap and
      row-gap around 24px) to lay out one section per curated article.
      Do NOT create a single "top story" block - turn every curated
      article into its own section: a bold uppercase heading (short,
      based on the article's own title/category) followed by a
      paragraph using that article's summary and key points as body
      text.
    - CRITICAL - EXACT ARTICLE COUNT, NO EMPTY OR BLANK CELLS: There are
      exactly {article_count} curated articles below - create EXACTLY
      that many sections, no more, no fewer, and no placeholder/empty
      sections. {"Since " + str(article_count) + " is odd, make ONLY the LAST article's section span both columns (grid-column: 1 / -1) so the grid ends cleanly with no empty trailing cell." if article_count % 2 == 1 else "Since " + str(article_count) + " is even, keep every section a single column so the grid fills perfectly with no empty trailing cell."}
      Never render an empty box, an empty <div>, or a solid-colored
      block with no real text in it anywhere on the page.
    - For 2-3 of the sections, add a short colored accent line
      (in a highlight color) above the paragraph showing that article's
      own category or relevance score.
    - At the bottom of each column, add one pale colored info box
      (light blue/light pink, or theme-appropriate pale color) containing
      that source article's category and a bold "Read More" link pointing
      to the real article url - only add this info box if there is a real
      article category and url to put inside it, never as an empty filler
      box. Use the real curated article titles, summaries, categories and
      urls throughout - never placeholder/lorem ipsum text.
    - Footer: a full-width colored strip at the very bottom with small
      bold centered white text reading something like "Compiled
      automatically by an AI pipeline | Generated on {current_date}".
    Give final response strictly in HTML only, no markdown, no code fences.

    Newsletter Title: {newsletter_title}
    Generated On: {current_date}
    Curated Summaries (use ALL and ONLY these, in this order): {curated_summaries}
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


def format_curated_summaries(curated_articles):
    """Formats the list of {title, url, summary_block} dicts into one
    text block for the HTML generator, and swaps in the real article
    count so the prompt's {{ARTICLE_COUNT}} placeholder is accurate."""

    blocks = []
    for idx, item in enumerate(curated_articles, start=1):
        blocks.append(
            f"Article {idx}:\n"
            f"Title: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"{item['summary_block']}\n"
        )
    return "\n".join(blocks)


# ========== DETERMINISTIC PIPELINE ===============
# Note: instead of letting an LLM "leader" agent freely decide when/how to
# call each tool, the pipeline below calls weekly_article_collector,
# article_summarizer, and newsletter_html_generator directly in a fixed
# order. This guarantees the country/region and category filters you pick
# are always respected exactly (no chance of the agent mixing in other
# countries), keeps the article count consistent with the slider so the
# HTML grid never has blank cells, and lets the UI show real progress for
# each stage instead of a single opaque spinner.

def run_newsletter_pipeline(country, category, max_results, newsletter_title, theme, status):
    # ---- Stage 1: Collect ----
    status.update(label=f"Collecting articles ({country} · {category})...", state="running")
    articles = weekly_article_collector(
        max_results=max_results, country=country, category=category
    )

    if not articles:
        status.update(label="No articles found.", state="error")
        return None

    st.write(f"Found {len(articles)} article(s).")

    # ---- Stage 2: Summarize each article ----
    status.update(label="Summarizing articles...", state="running")
    progress_bar = st.progress(0)
    curated = []
    for i, article in enumerate(articles, start=1):
        summary_block = article_summarizer(
            article_text=article.get("content") or article.get("title") or "",
            article_title=article.get("title") or "Untitled",
        )
        curated.append({
            "title": article.get("title") or "Untitled",
            "url": article.get("url") or "",
            "summary_block": summary_block,
        })
        progress_bar.progress(i / len(articles))

    # ---- Stage 3: Build HTML ----
    status.update(label="Building HTML newsletter...", state="running")
    curated_text = format_curated_summaries(curated)
    raw_html = newsletter_html_generator(
        curated_summaries=curated_text,
        newsletter_title=newsletter_title,
        theme=theme,
        article_count=len(curated),
    )
    html_code = raw_html.replace("```html", "").replace("```", "").strip()

    status.update(label="Newsletter ready!", state="complete")
    return html_code


# ========== RUN PIPELINE ===============
if st.button("Generate Newsletter"):
    with st.status("Starting newsletter generation...", expanded=True) as status:
        code = run_newsletter_pipeline(
            country=selected_country,
            category=selected_category,
            max_results=max_results,
            newsletter_title=newsletter_title,
            theme=selected_theme,
            status=status,
        )

    if code:
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
    else:
        st.error("Couldn't generate a newsletter — try a broader country/region or category.")
