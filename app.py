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


# =========== TOOL 1 ======================
def weekly_article_collector(max_results=5):
    """This function searches the web for the top trending news
    headlines published in the current week using the Tavily search
    API. Returns article metadata: title, url, content and
    published date."""

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
def newsletter_html_generator(curated_summaries, newsletter_title="Weekly Newsletter"):
    """This function converts curated article summaries
    into a styled html newsletter template suitable
    for email or web publishing, given curated summaries
    text and newsletter title"""

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
    1. Call the weekly_article_collector tool with max_results as given,
       to fetch general top trending news headlines across ANY subject
       for the week.
    2. Call the article_summarizer tool separately on EACH collected
       article to get its summary, key points, category and relevance
       score.
    3. 3. Keep the best EXACTLY 5 curated articles whenever 5 are available.
    4. Combine all the remaining curated summaries (title, summary,
       key points, category, url for each) into one collection, then
       call the newsletter_html_generator tool once with that full
       collection so all curated topics appear in the final newsletter.
    Give the final response output strictly in HTML, no markdowns,
    no code fences, no explanation text before or after the HTML.
    Instructions given below:
    """

    prompt = prompt + query

    response = agent.invoke({"messages": [{'role': 'user', 'content': prompt}]})
    code = _extract_text(response['messages'][-1])
    return code


# ========== CALLING MAIN AGENT ===============
if st.button("Generate Newsletter"):
    with st.spinner("Agent Running"):
        user_query = (
            f"Create this week's newsletter covering the top trending "
            f"news stories of the week across any topic/category "
            f"(do not restrict to a single subject)."
            + f"\nUse max_results={max_results} when collecting articles."
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
