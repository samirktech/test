
"""
AI Newsletter Generator - Streamlit App
Built using LangChain Agent (Tool Calling) + Gemini + Tavily
"""

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
from this week's top trending news using a LangChain agent.""")

st.sidebar.title("Fill Important Details")

# ============ API KEYS ===================
TAVILY_API_KEY = st.sidebar.text_input("Tavily-API", type="password")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini-API", type="password")

all_API = [TAVILY_API_KEY, GOOGLE_API_KEY]

if not all(all_API):
    st.error("Must give API keys")
    st.stop()

st.success("API KEYS LOADED SUCCESSFULLY")

# =========== MODEL CREATION ==============
model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    google_api_key=GOOGLE_API_KEY
)

max_results = 5

# ==================GET USER INFO=====================
st.markdown("### NEWSLETTER DETAILS")
newsletter_title = st.text_input("Newsletter Title", value="Weekly Digest")


# =========== TOOL 1 ======================
def weekly_article_collector(max_results=5):
    """Collect top trending world news of the week."""

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
            "published_date": result.get("published_date", "N/A"),
        })
    return articles


# =========== NEW TOOL : COUNTRY NEWS ======================
def top_country_news():
    """Fetch latest news for top 5 countries including India."""

    countries = [
        "India",
        "United States",
        "China",
        "United Kingdom",
        "Germany"
    ]

    client = TavilyClient(api_key=TAVILY_API_KEY)
    country_news = {}

    for country in countries:
        response = client.search(
            query=f"latest breaking news {country}",
            topic="news",
            time_range="day",
            max_results=5,
            include_answer=False,
        )

        articles = []
        for result in response.get("results", []):
            articles.append({
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
            })

        country_news[country] = articles

    return country_news


# =========== TOOL 2 ======================
def article_summarizer(article_text, article_title="Untitled"):
    """Summarize article using Gemini."""

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
def newsletter_html_generator(curated_summaries, newsletter_title="Weekly Newsletter"):
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


# ========== Agent Creation ================
agent = create_agent(
    model=model,
    tools=[weekly_article_collector, article_summarizer, newsletter_html_generator]
)


# ============== MAIN AGENT ===============
def main_agent(agent, query):

    prompt = """Your task is to orchestrate the full newsletter workflow:
    1. Collect weekly articles.
    2. Summarize each article.
    3. Keep best 5 articles.
    4. Generate one HTML newsletter from all curated summaries.
    Return only HTML.
    """

    response = agent.invoke(
        {"messages": [{'role': 'user', 'content': prompt + query}]}
    )

    return _extract_text(response['messages'][-1])


# ==========================================================
# NEW STREAMLIT SECTION : TOP 5 COUNTRY NEWS
# ==========================================================
st.divider()
st.subheader("🌍 Latest News From Top 5 Countries")

if st.button("Show Top 5 Countries News"):

    with st.spinner("Fetching country news..."):

        country_news = top_country_news()

        for country, articles in country_news.items():

            st.markdown(f"## {country}")

            if articles:
                for idx, article in enumerate(articles, start=1):

                    st.markdown(f"### {idx}. {article['title']}")

                    if article.get("content"):
                        st.write(article["content"])

                    if article.get("url"):
                        st.markdown(f"[Read full article]({article['url']})")

                    st.markdown("---")
            else:
                st.info("No recent news found.")


# ==========================================================
# EXISTING NEWSLETTER GENERATOR
# ==========================================================
st.divider()
st.subheader("📰 AI Newsletter Generator")

if st.button("Generate Newsletter"):

    with st.spinner("Agent Running..."):

        user_query = (
            f"Create this week's newsletter covering the top trending "
            f"news stories across any topic/category."
            + f"\nUse max_results={max_results}."
            + f"\nNewsletter Title: {newsletter_title}"
        )

        raw_code = main_agent(agent, user_query)
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

