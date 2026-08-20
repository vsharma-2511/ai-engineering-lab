from bs4 import BeautifulSoup
import httpx
from Research_Agent.src.tools.schemas import ExtractedContent, ScraperInput


def scrape_webpage(url: str, max_length: int = 4000) -> ExtractedContent:
    """Fetches a URL, strips HTML tags/scripts, and returns clean readable text."""
    print(f"🔧 [TOOL EXECUTED] scrape_webpage | url='{url}'")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(
            timeout=15.0, follow_redirects=True, headers=headers
        ) as client:
            response = client.get(url)

        status_code = response.status_code
        if status_code != 200:
            return ExtractedContent(
                url=url,
                title="Error",
                content=f"Failed to load page. HTTP status code: {status_code}",
                status_code=status_code,
            )

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, header, footer, and nav tags to reduce noise
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        title = soup.title.string.strip() if soup.title else "No Title"
        text = soup.get_text(separator="\n", strip=True)

        # Truncate text if it exceeds max_length
        cleaned_text = text[:max_length]
        if len(text) > max_length:
            cleaned_text += "\n...[Content Truncated]"

        return ExtractedContent(
            url=url,
            title=title,
            content=cleaned_text,
            status_code=status_code,
        )

    except Exception as e:
        return ExtractedContent(
            url=url,
            title="Exception",
            content=f"An error occurred while scraping: {str(e)}",
            status_code=500,
        )