"""
Tool for searching arXiv papers by subject classification category.

Searches papers within specific subject areas using arXiv's classification system.
"""

import json
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class ArxivSearchByCategoryTool(Tool):
    """
    Tool for searching arXiv papers by subject classification category.
    
    Description:
        Search papers within specific subject areas using arXiv's classification system.
        Supports primary and secondary category filtering.
    
    Input Parameters:
        - search_query (str, required): Category search query. Use format 'cat:category_code' for category search (e.g., 'cat:cs.AI', 'cat:math.CO')
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
        """Search arXiv papers by subject category."""
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
