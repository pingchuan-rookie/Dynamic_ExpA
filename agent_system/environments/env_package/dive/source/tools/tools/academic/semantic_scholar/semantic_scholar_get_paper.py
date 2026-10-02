"""
Tool for retrieving a specific paper from Semantic Scholar by ID.

Fetches complete paper information using various ID formats.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class SemanticScholarGetPaperTool(Tool):
    """
    Tool for retrieving a specific paper from Semantic Scholar by ID.
    
    Description:
        Fetches complete paper information using various ID formats.
        Supports Semantic Scholar ID, CorpusId, DOI, arXiv, MAG, ACL, PMID, PMCID, and URL formats.
    
    Input Parameters:
        - paper_id (str, required): Paper identifier in various formats:
          - Semantic Scholar ID: "649def34f8be52c8b66281af98ae884c09aef38b"
          - CorpusId: "CorpusId:215416146"
          - DOI: "DOI:10.18653/v1/N18-3011"
          - arXiv: "ARXIV:2106.15928"
          - MAG: "MAG:112218234"
          - ACL: "ACL:W12-3903"
          - PMID: "PMID:19872477"
          - PMCID: "PMCID:2323736"
          - URL: "URL:https://arxiv.org/abs/2106.15928v1"
        - fields (str, optional): Comma-separated list of fields to return
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON object containing paper details
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get paper details using Semantic Scholar API."""
        import time
        
        paper_id = params.get('paper_id')
        if not paper_id:
            return {"error": "paper_id is required"}
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        
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
