"""
AI Newsletter Generator
LangChain Agent + Gemini + Tavily + Streamlit
"""

import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Newsletter Generator",
    layout="wide"
)

st.title("AI NEWSLETTER GENERATOR")

st.write(
    "Build a curated, styled HTML newsletter from this week's "
    "top trending news using a LangChain agent."
)


# ============================================================
# SIDEBAR / API KEYS
# ============================================================

st.sidebar.title("Newsletter Settings")

TAVILY_API_KEY = st.sidebar.text_input(
    "Tavily API Key",
    type="password"
)

GOOGLE_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

if not TAVILY_API_KEY or not GOOGLE_API_KEY:
    st.info("Enter both API keys in the sidebar to continue.")
    st.stop()

st.success("API keys loaded successfully.")


# ============================================================
# MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# NEWSLETTER SETTINGS
# ============================================================

MAX_RESULTS = 8

st.markdown("### NEWSLETTER DETAILS")

newsletter_title = st.text_input(
    "Newsletter Title",
    value="Weekly Digest"
)


# ============================================================
# TOOL 1 — COLLECT NEWS
# ============================================================

def weekly_article_collector(max_results=MAX_RESULTS):
    """
    Search for the top trending news stories of the current week.
    Returns article title, URL, content and publication date.
    """

    query = "top trending world news headlines this week"

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
            "published_date": result.get(
                "published_date",
                "N/A"
            ),
        })

    return articles


# ============================================================
# TOOL 2 — SUMMARIZE ARTICLE
# ============================================================

def article_summarizer(
    article_text,
    article_title="Untitled"
):
    """
    Summarize an article and return:
    summary, key points, category and relevance score.
    """

    prompt = f"""
You are a professional newsletter editor.

Summarize the article below in 3-4 concise lines.
Then provide 2-3 key points, assign one category, and
give a relevance score out of 10 for a general audience.

Article Title:
{article_title}

Article Content:
{article_text}

Return ONLY this format:

Summary: <summary>
Key Points: <point1>; <point2>; <point3>
Category: <category>
Relevance: <score>/10
"""

    response = model.invoke(prompt)

    return _extract_text(response)


# ============================================================
# TOOL 3 — GENERATE NEWSLETTER HTML
# ============================================================

def newsletter_html_generator(
    curated_summaries,
    newsletter_title="Weekly Newsletter"
):
    """
    Convert curated article summaries into a complete,
    self-contained HTML newsletter.
    """

    current_date = datetime.datetime.now().strftime(
        "%d %B %Y"
    )

    prompt = f"""
Convert the curated article summaries below into a single
self-contained HTML newsletter.

Return a COMPLETE HTML document containing:
<html>, <head>, <style>, <body>.

GENERAL DESIGN:
- Printed magazine/newspaper newsletter appearance.
- Maximum content width around 900px.
- Centered page.
- Soft off-white background.
- Bordered page frame.
- Text-only design.
- Do NOT use images.

============================================================
CRITICAL TEXT VISIBILITY
============================================================

Every text element must have an explicit readable CSS color.

Set explicit color values on:
- html
- body
- headings
- paragraphs
- links
- spans
- list items
- metadata

Never rely on:
- inherit
- currentColor
- browser defaults

Every colored background must also have an explicit
text color with strong contrast.

============================================================
HEADER
============================================================

1. Top-left:
   Small bordered box containing exactly:
   "{current_date}"

2. Masthead:
   Large, bold, centered newsletter title:
   "{newsletter_title}"

3. Sub-banner:
   Full-width colored horizontal band below the masthead.

   Use a short uppercase tagline describing this week's
   collection of news.

   Do NOT assume the newsletter is only about AI or technology.

============================================================
ARTICLE GRID — VERY IMPORTANT
============================================================

Create exactly ONE grid container:

.news-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}

Every article MUST be contained inside exactly ONE:

<div class="article-card">

The following MUST remain INSIDE the same article-card:
- category
- article heading
- summary
- key points
- accent line
- relevance information
- Read More link
- source/category info box

NEVER create a separate grid item for:
- Read More
- category
- metadata
- accent line
- source information
- an empty div
- a filler box

============================================================
ODD / EVEN ARTICLE COUNTS
============================================================

This is critical.

Use this CSS:

.article-card:last-child:nth-child(odd) {{
    grid-column: 1 / -1;
}}

This means:

8 articles → normal 2-column grid
7 articles → last article spans both columns
6 articles → normal 2-column grid
5 articles → last article spans both columns
4 articles → normal 2-column grid

NEVER intentionally create an empty grid cell.

NEVER add a blank article or placeholder just to balance
the grid.

Do not manually add unnecessary grid-column rules to
individual articles.

============================================================
ARTICLE CONTENT
============================================================

Every curated article becomes its own article-card.

Each card should contain:

- Bold uppercase heading
- Article summary
- 2-3 key points where appropriate
- Category
- Relevance score where appropriate
- Real article URL
- Bold "Read More" link

Use the real data from the curated summaries.

Never use:
- lorem ipsum
- placeholder articles
- fake URLs
- empty sections

For 2-3 articles, add a short colored accent line above
the article content.

============================================================
ARTICLE INFO BOX
============================================================

Inside each article-card, add a pale colored info box
containing:

Category: <category> | Read More

The Read More link MUST point to the actual article URL.

Only create this box when both category and URL are valid.

The info box MUST NOT become its own grid item.

============================================================
FOOTER
============================================================

At the bottom create a full-width colored footer.

Use small, bold, centered white text similar to:

Compiled automatically by a multi-agent AI pipeline |
Generated on {current_date}

============================================================
HTML OUTPUT RULE
============================================================

Return ONLY HTML.

Do not return:
- Markdown
- Code fences
- Explanations
- ```html
- ``` 

Newsletter Title:
{newsletter_title}

Generated On:
{current_date}

Curated Summaries:
{curated_summaries}
"""

    response = model.invoke(prompt)

    html = _extract_text(response)

    return _clean_html_output(html)


# ============================================================
# HELPER — EXTRACT MODEL TEXT
# ============================================================

def _extract_text(response):
    """
    Safely extract text from a LangChain model response.
    """

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            else:
                text_parts.append(str(block))

        return "".join(text_parts)

    return str(content)


# ============================================================
# HELPER — CLEAN GENERATED HTML
# ============================================================

def _clean_html_output(html):
    """
    Remove accidental Markdown code fences and surrounding
    whitespace from the model-generated HTML.
    """

    html = html.strip()

    if html.startswith("```html"):
        html = html[len("```html"):].strip()

    elif html.startswith("```HTML"):
        html = html[len("```HTML"):].strip()

    if html.endswith("```"):
        html = html[:-3].strip()

    return html


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=[
        weekly_article_collector,
        article_summarizer,
        newsletter_html_generator
    ]
)


# ============================================================
# MAIN AGENT
# ============================================================

def main_agent(agent, query):
    """
    Orchestrate the complete newsletter workflow.
    """

    prompt = f"""
Your task is to create the complete weekly newsletter.

Follow these steps in order:

1. Call weekly_article_collector with:
   max_results={MAX_RESULTS}

2. Call article_summarizer separately for EACH collected article.

3. Select the best EXACTLY {MAX_RESULTS} articles whenever
   {MAX_RESULTS} or more valid articles are available.

4. Preserve each selected article's:
   - title
   - summary
   - key points
   - category
   - relevance score
   - URL

5. Combine the selected article information into ONE collection.

6. Call newsletter_html_generator EXACTLY ONCE with the
   complete collection.

7. Return ONLY the final HTML newsletter.

Do NOT return:
- Markdown
- code fences
- explanations
- commentary before or after the HTML

User request:
{query}
"""

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    final_message = response["messages"][-1]

    return _clean_html_output(
        _extract_text(final_message)
    )


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

if st.button(
    "Generate Newsletter",
    type="primary"
):

    try:

        with st.spinner(
            "Collecting news and generating newsletter..."
        ):

            user_query = (
                "Create this week's newsletter covering "
                "the top trending news stories across "
                "any topic/category. "
                f"Use max_results={MAX_RESULTS}. "
                f"Newsletter Title: {newsletter_title}"
            )

            code = main_agent(
                agent,
                user_query
            )

        if not code or "<html" not in code.lower():
            st.error(
                "The generated response does not appear "
                "to contain valid HTML."
            )
            st.stop()

        st.success("Newsletter generated successfully!")

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.download_button(
            label="Download newsletter.html",
            data=code,
            file_name="newsletter.html",
            mime="text/html"
        )

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        st.divider()

        st.subheader("Preview")

        st.components.v1.html(
            code,
            height=900,
            scrolling=True
        )

    except Exception as e:

        st.error(
            "An error occurred while generating the newsletter."
        )

        st.exception(e)
