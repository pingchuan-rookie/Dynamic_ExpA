"""
Tool for retrieving a specific author from Semantic Scholar by ID.

Fetches complete author information using author ID.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class SemanticScholarGetAuthorTool(Tool):
    """
    Tool for retrieving a specific author from Semantic Scholar by ID.
    
    Description:
        Fetches complete author information using author ID.
        Supports various author ID formats and field selection.
    
    Input Parameters:
        - author_id (str, required): Author identifier
        - fields (str, optional): Comma-separated list of fields to return
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON object containing author details
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get author details using Semantic Scholar API."""
        import time
        
        author_id = params.get('author_id')
        if not author_id:
            return {"error": "author_id is required"}
        base_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
        
        if params:
            url = f"{base_url}?{urlencode(params)}"
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
                    base_delay = 1 + (attempt * 10)
                    delay = base_delay + random.uniform(0, base_delay * 0.5)
                    time.sleep(delay)
                    continue
        return {"error": "Max retries exceeded"}
