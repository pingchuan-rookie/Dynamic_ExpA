"""
Tool for filtering prescribable Rxnorm concepts by property values.

This tool returns concepts that match specified property criteria.
"""

import json
from typing import Any, Dict, List
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class PrescribableFilterByPropertyTool(Tool):
    """
    Tool for filtering prescribable Rxnorm concepts by property values.
    
    Description:
        Filter prescribable Rxnorm concepts to check if they match specified property criteria.
        Uses the RxNAV REST API to validate property values for a given RxCUI.
    
    Input Parameters:
        - rxcui (str, required): Rxnorm Concept Unique Identifier
        - propName (str, required): Property name to filter by (e.g., "AVAILABLE_STRENGTH", "has_tradename")
        - propValues (str, optional): Specific property values to match against
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON object containing the filtered concept information
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            rxcui = params.get("rxcui", "")
            base_url = f"https://rxnav.nlm.nih.gov/REST/Prescribe/rxcui/{rxcui}/filter.json"
            
            query_params = {k: v for k, v in params.items() if k != "rxcui" and v}
            url = f"{base_url}?{urlencode(query_params)}"
            
            if hasattr(context, 'logger') and context.logger:
                context.logger.info(f"Making request to: {url}")
            
            with urlopen(url) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}
