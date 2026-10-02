"""
Tool for paper title search using Semantic Scholar API.

Search for a single paper based on closest title match to given query.
Returns the best matching paper with match score.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class SemanticScholarPaperTitleSearchTool(Tool):
    """
    Tool for paper title search using Semantic Scholar API.
    
    Description:
        Search for a single paper based on closest title match to given query.
        Returns the best matching paper with match score.
        Uses plain-text search without special query syntax.
    
    Input Parameters:
        - query (str, required): Plain-text search query string
        - fields (str, optional): Comma-separated list of fields to return
        - publicationTypes (str, optional): Filter by publication types
        - openAccessPdf (str, optional): Filter for papers with public PDF
        - minCitationCount (str, optional): Minimum number of citations
        - publicationDateOrYear (str, optional): Publication date range
        - year (str, optional): Publication year range
        - venue (str, optional): Publication venues
        - fieldsOfStudy (str, optional): Fields of study
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON object with data array containing best match
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for paper by title match using Semantic Scholar API."""
        import time
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
        url = f"{base_url}?{urlencode(params)}"
        
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
