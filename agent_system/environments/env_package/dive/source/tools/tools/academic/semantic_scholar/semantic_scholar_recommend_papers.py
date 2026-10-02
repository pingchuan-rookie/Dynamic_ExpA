"""
Tool for paper recommendations using Semantic Scholar API.

Get paper recommendations based on a list of papers.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class SemanticScholarRecommendPapersTool(Tool):
    """
    Tool for paper recommendations using Semantic Scholar API.
    
    Description:
        Get paper recommendations based on a list of papers.
        Uses collaborative filtering and content-based recommendations.
    
    Input Parameters:
        - papers (list, required): List of paper IDs to base recommendations on
        - limit (int, optional): Maximum number of recommendations (default: 10)
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON object with recommended papers
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get paper recommendations using Semantic Scholar API."""
        import time
        
        papers = params.get('papers', [])
        if not papers:
            return {"error": "No papers provided"}
        
        paper_id = papers[0]
        base_url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
        
        query_params = {}
        if 'limit' in params:
            query_params['limit'] = params['limit']
        
        if query_params:
            url = f"{base_url}?{urlencode(query_params)}"
        else:
            url = base_url
        
        headers = {
            'User-Agent': 'VerifiableTools/1.0',
            'Accept': 'application/json'
        }
        
        if context.auth and 'api_key' in context.auth:
            headers['x-api-key'] = context.auth['api_key']
        
        request = Request(url, headers=headers)
        
        import random
        time.sleep(random.uniform(1, 100))
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with urlopen(request) as response:
                    content = response.read().decode('utf-8')
                    return json.loads(content)
                    
            except HTTPError as e:
                if attempt == max_retries - 1:
                    return {"error": f"HTTP {e.code}: {e.reason} (after {max_retries} attempts)"}
                else:
                    base_delay = 1 + (attempt * 10)
                    delay = base_delay + random.uniform(0, base_delay * 0.5)
                    time.sleep(delay)
                    continue
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"{str(e)} (after {max_retries} attempts)"}
                else:
                    base_delay = 1 + (attempt * 2)
                    delay = base_delay + random.uniform(0, base_delay * 0.5)
                    time.sleep(delay)
                    continue
        return {"error": "Max retries exceeded"}
