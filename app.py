import datetime

import streamlit as st
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Newsletter Generator",
    page_icon="📰",
    layout="wide"
)

st.title("AI NEWSLETTER GENERATOR")

st.write(
    "Build a curated, styled HTML newsletter from this week's "
    "top trending news using a LangChain agent."
)


# ============================================================
# SIDEBAR
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

newsletter_title = st.text_input(
    "Newsletter Title",
    value="Weekly Digest"
)

MAX_RESULTS = 8


# ============================================================
# CHECK API KEYS
# ============================================================

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
# HELPER
# ============================================================

def extract_text(response):
    """Extract text safely from a LangChain response."""

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))

        return "".join(parts)

    return str(content)


def clean_html(html):
    """Remove accidental Markdown code fences."""

    html = html.strip()

    if html.startswith("```html"):
        html = html[7:].strip()

    elif html.startswith("```HTML"):
        html = html[7:].strip()

    if html.endswith("```"):
        html = html[:-3].strip()

    return html


# ============================================================
# TOOL 1 — COLLECT ARTICLES
# ============================================================

def weekly_article_collector(max_results=8):
    """
    Search for trending news from the current week.
    """

    client = TavilyClient(
        api_key=TAVILY_API_KEY
    )

    response = client.search(
        query="top trending world news headlines this week",
        topic="news",
        time_range="week",
        max_results=max_results,
        include_answer=False
    )

    articles = []

    for result in response.get("results", []):
        articles.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "published_date": result.get(
                "published_date",
                "N/A"
            )
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
    Summarize one article.
    """

    prompt = f"""
You are a professional newsletter editor.

Summarize the article below in 3-4 concise lines.

Then provide:
- 2-3 key points
- One category
- Relevance score out of 10

Article Title:
{article_title}

Article Content:
{article_text}

Return ONLY this format:

Summary: <summary>
Key Points: <point1>; <point2>; <point3>
Category: <Tech/Business/Science/World/Other>
Relevance: <score>/10
"""

    response = model.invoke(prompt)

    return extract_text(response)


# ============================================================
# TOOL 3 — GENERATE NEWSLETTER
# ============================================================

def newsletter_html_generator(
    curated_summaries,
    newsletter_title="Weekly Newsletter"
):
    """
    Generate the final HTML newsletter.
    """

    current_date = datetime.datetime.now().strftime(
        "%d %B %Y"
    )

    prompt = f"""
Create a complete, self-contained HTML newsletter.

Return ONLY HTML.
Do not use Markdown.
Do not use code fences.

Newsletter title:
{newsletter_title}

Generation date:
{current_date}

Curated articles:
{curated_summaries}


========================
DESIGN
========================

Create a newspaper/magazine-style newsletter.

Requirements:

- Maximum content width: approximately 900px.
- Center the newsletter.
- Use an off-white page background.
- Use a visible page border.
- Use text and colored boxes only.
- Do NOT use images.
- Do NOT use image URLs.
- Do NOT use background images.


========================
TEXT VISIBILITY
========================

Every text element MUST have an explicit CSS color.

Explicitly set colors for:

html
body
headings
paragraphs
links
spans
list items

Never use:

inherit
currentColor

Every colored background must have an explicit
background-color and a contrasting text color.


========================
HEADER
========================

At the top-left:

Small bordered box containing:

{current_date}


Then create a large centered masthead:

{newsletter_title}


Under it create a full-width colored banner.

The banner should contain a short uppercase tagline
describing the week's news.

Do not assume the newsletter only contains technology
or AI news.


========================
ARTICLE GRID
========================

Create exactly ONE grid container:

.news-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}


Each article MUST be exactly ONE:

<div class="article-card">

The article card must contain:

- Category
- Heading
- Summary
- Key points
- Relevance score when available
- Accent line when appropriate
- Category information
- Read More link


IMPORTANT:

The category/read-more box MUST remain INSIDE
the article-card.

Never create separate grid items for:

- Category
- Read More
- Metadata
- Accent lines
- Source information
- Empty divs
- Placeholder boxes


========================
NO EMPTY GRID CELLS
========================

Use this CSS:

.article-card:last-child:nth-child(odd) {{
    grid-column: 1 / -1;
}}

This means:

8 articles = 2 columns
7 articles = final article spans both columns
6 articles = 2 columns
5 articles = final article spans both columns
4 articles = 2 columns

Never create an empty article.

Never create a placeholder.

Never create an empty div just to balance
the grid.

Do NOT manually force individual articles
to span columns.


========================
ARTICLE CONTENT
========================

Every curated article must appear.

Use the real:

- Article title
- Summary
- Key points
- Category
- Relevance
- URL

Do not use:

- Lorem ipsum
- Fake URLs
- Placeholder content
- Empty sections


========================
READ MORE BOX
========================

Inside each article-card, create a pale-colored
information box:

Category: <category> | Read More

"Read More" must link to the real article URL.

Only create this box when the category and URL exist.

The box must NOT become another grid item.


========================
FOOTER
========================

Create a full-width footer at the bottom.

Use small, bold, centered white text similar to:

Compiled automatically by a multi-agent AI pipeline |
Generated on {current_date}


========================
FINAL RULE
========================

Return ONLY the final HTML document.

No Markdown.
No ```html.
No ``` .
No explanation before or after the HTML.
"""

    response = model.invoke(prompt)

    return clean_html(
        extract_text(response)
    )


# ============================================================
# CREATE AGENT
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

def main_agent(query):
    """
    Orchestrate the newsletter workflow.
    """

    prompt = f"""
Create the weekly newsletter by following these steps:

1. Call weekly_article_collector with:
   max_results={MAX_RESULTS}

2. Call article_summarizer separately for every
   collected article.

3. Select the best {MAX_RESULTS} articles whenever
   at least {MAX_RESULTS} valid articles are available.

4. Preserve for every selected article:

   title
   summary
   key points
   category
   relevance
   URL

5. Combine all selected article information.

6. Call newsletter_html_generator once with the
   complete collection.

7. Return only the final HTML.

Do not return Markdown.
Do not return code fences.
Do not return explanations.

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

    return clean_html(
        extract_text(final_message)
    )


# ============================================================
# GENERATE NEWSLETTER
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
                "any topic or category. "
                f"Use max_results={MAX_RESULTS}. "
                f"Newsletter Title: {newsletter_title}"
            )

            newsletter_html = main_agent(
                user_query
            )

        if not newsletter_html:
            st.error(
                "No newsletter was generated."
            )
            st.stop()

        if "<html" not in newsletter_html.lower():
            st.error(
                "The generated response is not valid HTML."
            )
            st.code(newsletter_html)
            st.stop()

        st.success(
            "Newsletter generated successfully!"
        )

        # Download
        st.download_button(
            label="Download newsletter.html",
            data=newsletter_html,
            file_name="newsletter.html",
            mime="text/html"
        )

        st.divider()

        # Preview
        st.subheader("Preview")

        st.components.v1.html(
            newsletter_html,
            height=900,
            scrolling=True
        )

    except Exception as error:

        st.error(
            "An error occurred while generating "
            "the newsletter."
        )

        st.exception(error)
