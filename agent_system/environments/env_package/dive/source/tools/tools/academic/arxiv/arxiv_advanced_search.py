"""
Tool for advanced arXiv paper search with complex query syntax.

This tool supports complex queries with multiple field prefixes,
boolean operators (AND, OR, ANDNOT), and advanced search patterns.
"""

import json
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class ArxivAdvancedSearchTool(Tool):
    """
    Tool for advanced arXiv paper search with complex query syntax.
    
    Description:
        Search arXiv papers using advanced query syntax with Boolean operators,
        field-specific searches, and sorting options.
    
    Input Parameters:
        - search_query (str, required): Advanced search query with arXiv syntax
        - start (int, optional): Starting index for results (default: 0)
        - max_results (int, optional): Maximum number of results (default: 10)
        - sortBy (str, optional): Sort by relevance, lastUpdatedDate, submittedDate
        - sortOrder (str, optional): ascending or descending
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: XML string containing search results
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced arXiv search with complex boolean queries."""
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
