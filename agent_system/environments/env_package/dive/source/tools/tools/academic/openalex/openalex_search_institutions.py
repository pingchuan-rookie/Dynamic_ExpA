"""
Tool for searching institutions on OpenAlex.

Basic search functionality for academic institutions and organizations.
"""

import json
import time
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class OpenalexSearchInstitutionsTool(Tool):
    """
    Tool for searching institutions on OpenAlex.
    
    Description:
        Search for academic institutions and organizations.
        Searches through institution names, acronyms, and alternative names.
    
    Input Parameters:
        - search (str, optional): Search query string for institution names
        - filter (str, optional): Filter criteria (e.g., "country_code:us", "type:education")
        - sort (str, optional): Sort order (e.g., "works_count:desc", "cited_by_count:desc")
        - per-page (int, optional): Number of results per page (default: 25)
        - page (int, optional): Page number for pagination (default: 1)
        - select (str, optional): Comma-separated list of fields to return
        - mailto (str, optional): Email for polite pool (higher rate limits)
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON string containing institutions data
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search OpenAlex institutions."""
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                base_url = "https://api.openalex.org/institutions"
                url = f"{base_url}?{urlencode(params)}"
                
                with urlopen(url) as response:
                    return response.read().decode('utf-8')
                    
            except HTTPError as e:
                if attempt == max_retries - 1:
                    return {"error": f"HTTP {e.code}: {e.reason}"}
                else:
                    time.sleep(5 * (attempt + 1))
                    continue
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": str(e)}
                else:
                    time.sleep(5 * (attempt + 1))
                    continue
        
        return {"error": "Max retries exceeded"}
