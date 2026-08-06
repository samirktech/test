"""
AI Newsletter Generator - Streamlit App
Built using LangChain Agent (Tool Calling) + Gemini + Tavily
"""

import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")

# ================= TITLE =======================
st.title("AI NEWSLETTER GENERATOR")

st.write(
    """
    This app helps you build a curated, styled HTML newsletter
    from this week's top trending news using a LangChain agent.
    """
)

# ================= SIDEBAR =====================
st.sidebar.title("Fill Important Details")

TAVILY_API_KEY = st.sidebar.text_input("Tavily-API", type="password")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini-API", type="password")

if not all([TAVILY_API_KEY, GOOGLE_API_KEY]):
    st.error("Must give API keys")
    st.stop()

st.success("API KEYS LOADED SUCCESSFULLY")

# ================= MODEL =======================
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
)

max_results = 5

# ================= USER INPUT ==================
st.markdown("### NEWSLETTER DETAILS")

newsletter_title = st.text_input(
    "Newsletter Title",
    value="Weekly Digest"
)

# NEW: Country / Region selector
news_scope = st.selectbox(
    "Select News Region",
    [
        "Whole World",
        "India",
        "China",
        "America",
        "Britain",
        "France"
    ]
)

# ================= TOOL 1 ======================
def weekly_article_collector(region="Whole World", max_results=5):
    """Collect top trending news for selected region."""

    query_map = {
        "Whole World": "top trending world news headlines this week",
        "India": "top trending India news headlines this week",
        "China": "top trending China news headlines this week",
        "America": "top trending United States news headlines this week",
        "Britain": "top trending United Kingdom news headlines this week",
        "France": "top trending France news headlines this week",
    }

    query = query_map.get(region, query_map["Whole World"])

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

# ================= TOOL 2 ======================
def article_summarizer(article_text, article_title="Untitled"):
    """Summarize article using Gemini."""

    prompt = f"""
    You are a professional newsletter editor.

    Summarize the article below in 3-4 concise lines,
    then list 2-3 key points as bullets, assign a single
    category (Tech/Business/Science/World/Other) and give
    a relevance score out of 10 for a general audience.

    Article Title: {article_title}

    Article Content:
    {article_text}

    Give output strictly in this format:

    Summary: <summary>
    Key Points: <point1>; <point2>; <point3>
    Category: <category>
    Relevance: <score>/10
    """

    response = model.invoke(prompt)
    return _extract_text(response)

# ================= TOOL 3 ======================
def newsletter_html_generator(
    curated_summaries,
    newsletter_title="Weekly Newsletter"
):
    """Generate HTML newsletter."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")

    prompt = f"""
    Create a clean professional HTML newsletter.

    Newsletter Title: {newsletter_title}
    Generated On: {current_date}

    Curated Summaries:
    {curated_summaries}

    Return only valid HTML.
    """

    response = model.invoke(prompt)
    return _extract_text(response)

# ================= HELPER ======================
def _extract_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list) and content:
        last = content[-1]

        if isinstance(last, dict) and "text" in last:
            return last["text"]

        return str(last)

    return str(content)

# ================= AGENT =======================
agent = create_agent(
    model=model,
    tools=[
        weekly_article_collector,
        article_summarizer,
        newsletter_html_generator
    ]
)

# ================= MAIN AGENT =================
def main_agent(agent, query):

    prompt = """
    Your task is to orchestrate the full newsletter workflow:

    1. Collect weekly articles for the user-selected region or country.
    2. Summarize each article.
    3. Keep the best 5 articles.
    4. Generate one HTML newsletter from all curated summaries.

    Return only HTML.
    """

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt + query}]}
    )

    return _extract_text(response["messages"][-1])

# ==========================================================
# NEWSLETTER GENERATOR
# ==========================================================
st.divider()
st.subheader("📰 AI Newsletter Generator")

if st.button("Generate Newsletter"):

    with st.spinner("Agent Running..."):

        user_query = (
            f"Create this week's newsletter for: {news_scope}. "
            f"Cover the top trending news stories from that region only. "
            f"If the region is Whole World, include global news."
            + f"\nUse max_results={max_results}."
            + f"\nNewsletter Title: {newsletter_title}"
        )

        raw_code = main_agent(agent, user_query)

        code = (
            raw_code
            .replace("```html", "")
            .replace("```", "")
            .strip()
        )

        st.success(f"Newsletter generated for {news_scope}!")

        st.download_button(
            "Download newsletter.html",
            data=code,
            file_name="newsletter.html",
            mime="text/html",
        )

        st.divider()
        st.subheader("Preview")

        st.components.v1.html(
            code,
            height=900,
            scrolling=True
        )
