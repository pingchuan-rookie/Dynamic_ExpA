"""
Tool for searching arXiv papers by author name.

Searches for papers authored by specific individuals using author field queries.
"""

import json
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class ArxivSearchByAuthorTool(Tool):
    """
    Tool for searching arXiv papers by author name.
    
    Description:
        Search arXiv papers authored by specific individuals using author field queries.
        Supports partial name matches and multiple author formats.
    
    Input Parameters:
        - search_query (str, required): Author search query. Use format 'au:author_name' for author search (e.g., 'au:del_maestro')
        - start (int, optional): Starting index for results (default: 0)
        - max_results (int, optional): Maximum number of results (default: 10)
        - sortBy (str, optional): Sort by lastUpdatedDate, submittedDate, relevance
        - sortOrder (str, optional): ascending or descending
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: XML string containing search results
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search arXiv papers by author name."""
        import time
        import random
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                base_url = "https://export.arxiv.org/api/query"
                url = f"{base_url}?{urlencode(params)}"
                
                with urlopen(url) as response:
                        content = response.read().decode('utf-8')
                        return content
                    
            except HTTPError as e:
                if attempt == max_retries - 1:
                    return {"error": f"HTTP {e.code}: {e.reason}"}
                random_delay = base_delay + random.uniform(0, 2) + (attempt * 0.5)
                print(f"HTTP错误，第{attempt + 1}次重试，延迟{random_delay:.2f}秒: {e}")
                time.sleep(random_delay)
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": str(e)}
                random_delay = base_delay + random.uniform(0, 2) + (attempt * 0.5)
                time.sleep(random_delay)
        return {"error": "Max retries exceeded"}
