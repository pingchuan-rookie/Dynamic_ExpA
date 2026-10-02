"""
Tool for retrieving a specific source from OpenAlex by ID.

Fetches complete source information using OpenAlex identifiers.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class OpenalexGetSourceTool(Tool):
    """
    Tool for retrieving sources from OpenAlex.
    
    Description:
        Access OpenAlex sources API endpoint directly.
        All parameters are passed through to the API without modification.
    
    Input Parameters:
        All parameters are passed directly to the OpenAlex API.
        Common parameters include filters, search queries, pagination, etc.
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON string containing API response
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
        
    Design Philosophy:
        Pure pass-through - API returns what, the tool returns directly.
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve OpenAlex source by ID."""
        import time
        from urllib.request import Request
        
        source_id = params.get('source_id')
        if not source_id:
            return {"error": "'source_id' is required"}
        
        base_url = f"https://api.openalex.org/sources/{source_id}"
        
        other_params = {k: v for k, v in params.items() if k != 'source_id'}
        if other_params:
            url = f"{base_url}?{urlencode(other_params)}"
        else:
            url = base_url
        
        headers = {
            'User-Agent': 'VerifiableTools/1.0 (https://github.com/your-org/verifiable-tools)',
            'Accept': 'application/json'
        }
        
        request = Request(url, headers=headers)
        
        import random
        time.sleep(random.uniform(1, 3))
        
        import os
        proxy_handler = None
        if os.environ.get('http_proxy') or os.environ.get('https_proxy'):
            from urllib.request import ProxyHandler, build_opener
            proxy_handler = ProxyHandler({
                'http': os.environ.get('http_proxy', ''),
                'https': os.environ.get('https_proxy', '')
            })
            opener = build_opener(proxy_handler)
        else:
            opener = urlopen
        
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                if proxy_handler:
                    with opener.open(request) as response:
                        content = response.read().decode('utf-8')
                        return json.loads(content)
                else:
                    with urlopen(request) as response:
                        content = response.read().decode('utf-8')
                        return json.loads(content)
                    
            except HTTPError as e:
                if attempt == max_retries - 1:
                    return {"error": f"HTTP {e.code}: {e.reason} (after {max_retries} attempts)"}
                else:
                    base_delay = 1 + (attempt * 2)
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
