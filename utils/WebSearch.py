import os
from typing import Optional
from agno.tools import tool
from firecrawl import FirecrawlApp
from exa_py import Exa


@tool
async def web_search(query: str, search_type: str = "auto", max_results: int = 5) -> str:
    """
    Searches the web using the Exa AI Search API, with a fallback to Firecrawl Search.

    Args:
        query: The search query string (e.g., "latest advancements in solid state batteries").
        search_type: The search modality for Exa. Options include "auto", "instant", "fast",
                     "deep-lite", "deep", and "deep-reasoning".
        max_results: The maximum number of search results to return (default 5).

    Returns:
        A formatted string containing the title, URL, and snippets/highlights for each result,
        or an aggregated error message if both search providers fail.
    """
    errors = []


    if Exa is not None:
        try:
            exa = Exa()
            response = exa.search(
                query,
                type=search_type,
                num_results=max_results,
                contents={"highlights": True}
            )

            results = getattr(response, "results", [])
            if results:
                formatted_results = ["[Source: Exa Search]"]
                for idx, res in enumerate(results, start=1):
                    title = getattr(res, "title", "No Title")
                    url = getattr(res, "url", "Unknown URL")

                    highlights = getattr(res, "highlights", [])
                    snippet = " ".join(highlights) if highlights else "No snippet available."

                    formatted_results.append(
                        f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}\n{'-' * 40}"
                    )
                return "\n".join(formatted_results)
            else:
                return f"No results found for query: '{query}' via Exa."

        except Exception as e:
            errors.append(f"Exa API Error: {str(e)}")
    else:
        errors.append("Exa library ('exa-py') is not installed.")


    if FirecrawlApp is not None:
        try:
            app = FirecrawlApp()
            response = app.search(
                query=query,
                params={
                    "limit": max_results,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                        "onlyMainContent": True
                    }
                }
            )

            data = response.get("data", [])
            if data:
                formatted_results = ["[Source: Firecrawl Search Fallback]"]
                for idx, res in enumerate(data, start=1):
                    title = res.get("title", "No Title")
                    url = res.get("url", "Unknown URL")


                    desc = res.get("description", "")
                    markdown_content = res.get("markdown", "")
                    snippet = desc if desc else (markdown_content[:300] + "..." if markdown_content else "No snippet available.")

                    formatted_results.append(
                        f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}\n{'-' * 40}"
                    )
                return "\n".join(formatted_results)
            else:
                return f"No results found for query: '{query}' via Firecrawl."

        except Exception as e:
            errors.append(f"Firecrawl API Error: {str(e)}")
    else:
        errors.append("Firecrawl library ('firecrawl-py') is not installed.")

    return "Web search completely failed. Diagnostic details:\n" + "\n".join(errors)
