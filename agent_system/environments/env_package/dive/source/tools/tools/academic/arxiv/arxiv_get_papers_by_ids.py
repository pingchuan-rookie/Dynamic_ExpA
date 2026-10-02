"""
Tool for retrieving specific arXiv papers by their IDs.

Fetches complete paper information using arXiv identifiers.
"""

import json
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class ArxivGetPapersByIdsTool(Tool):
    """
    Tool for retrieving specific arXiv papers by their IDs.
    
    Description:
        Fetch complete paper information using arXiv identifiers.
        Supports both old-style and new-style arXiv IDs.
    
    Input Parameters:
        - id_list (str, required): Comma-separated list of arXiv paper IDs
        - start (int, optional): Starting index for results (default: 0)
        - max_results (int, optional): Maximum number of results (default: 10)
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: XML string containing paper details
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve arXiv papers by their IDs."""
        import time
        
        base_url = "https://export.arxiv.org/api/query"
        url = f"{base_url}?{urlencode(params)}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urlopen(url) as response:
                    content = response.read().decode('utf-8')
                    return content
                    
            except HTTPError as e:
                if attempt == max_retries - 1:
                    return {"error": f"HTTP {e.code}: {e.reason} (after {max_retries} attempts)"}
                else:
                    time.sleep(1 * (attempt + 1))
                    continue
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"{str(e)} (after {max_retries} attempts)"}
                else:
                    time.sleep(1 * (attempt + 1))
                    continue
        return {"error": "Max retries exceeded"}
