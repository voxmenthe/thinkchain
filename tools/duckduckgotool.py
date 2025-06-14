from tools.base import BaseTool
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

class DuckduckgoTool(BaseTool):
    name = "duckduckgotool"
    description = '''
    Searches the internet using DuckDuckGo and returns relevant results with titles, 
    descriptions, and URLs. Perfect for finding current information, news, restaurants, 
    businesses, or any web content.
    
    Use this tool when users ask about:
    - Current information not in your training data
    - Restaurant recommendations in specific locations
    - Business listings or contact information  
    - Recent news or events
    - "Search for [anything]" or "Find information about [topic]"
    '''
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 8)", 
                "default": 8
            }
        },
        "required": ["query"]
    }

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        num_results = kwargs.get("num_results", 8)

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select('.result')[:num_results]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                url_elem = result.select_one('.result__url')
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    url = url_elem.get('href') if url_elem else None
                    
                    results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n")

            if not results:
                return "No results found."
                
            return "\n".join(results)

        except requests.RequestException as e:
            return f"Error performing search: {str(e)}"