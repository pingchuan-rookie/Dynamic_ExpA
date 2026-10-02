"""
Tool for batch retrieval of authors from Semantic Scholar by IDs.

Get details for multiple authors at once using POST request.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class SemanticScholarAuthorBatchTool(Tool):
    """
    Tool for batch retrieval of authors from Semantic Scholar by IDs.
    
    Description:
        Get details for multiple authors at once using POST request.
        Supports up to 1,000 author IDs per request.
        Fields parameter is passed as query parameter, not in POST body.
    
    Input Parameters:
        - ids (list, required): List of author IDs to retrieve
        - fields (str, optional): Comma-separated list of fields to return
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON array containing author details
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Batch retrieve authors using Semantic Scholar API."""
        import time
        
        base_url = "https://api.semanticscholar.org/graph/v1/author/batch"
        
        ids = params.get('ids', [])
        fields = params.get('fields')
        
        query_params = {}
        if fields:
            query_params['fields'] = fields
        
        if query_params:
            url = f"{base_url}?{urlencode(query_params)}"
        else:
            url = base_url
        
        data = {"ids": ids}
        json_data = json.dumps(data).encode('utf-8')
        
        headers = {
            'User-Agent': 'VerifiableTools/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if context.auth and 'api_key' in context.auth:
            headers['x-api-key'] = context.auth['api_key']
        
        request = Request(url, data=json_data, headers=headers, method='POST')
        
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
