"""
Tool for getting multi-ingredient brand drug concepts from Rxnorm.

This tool retrieves multi-ingredient brand information for drug concepts.
"""

import json
from typing import Any, Dict
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class RxnormGetMultiIngredBrandTool(Tool):
    """
    Tool for getting multi-ingredient brand drug concepts from Rxnorm.
    
    Description:
        Retrieve multi-ingredient brand information for drug concepts using the RxNAV REST API.
        Returns brand names that contain multiple active ingredients.
    
    Input Parameters:
        - ingredientids (str, required): Space-separated list of ingredient RxCUIs (e.g., '161+1191')
        - sources (str, optional): Source vocabularies to search
    
    Output Format:
        Returns the raw API response directly without modification.
        - Success: JSON object containing multi-ingredient brand information
        - Error: {"error": "HTTP error message"} or {"error": "exception message"}
    
    Design Philosophy:
        Complete API pass-through - no internal validation, direct parameter forwarding to API.
    """

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get multi-ingredient brand drug concepts from Rxnorm."""
        try:
            base_url = "https://rxnav.nlm.nih.gov/REST/brands.json"
            
            if "ingredientids" in params and params["ingredientids"]:
                ingredientids = params["ingredientids"]
                other_params = {k: v for k, v in params.items() if k != "ingredientids" and v is not None}
                
                url = f"{base_url}?ingredientids={ingredientids}"
                if other_params:
                    url += "&" + urlencode(other_params)
            else:
                query_params = {k: v for k, v in params.items() if v is not None}
                url = f"{base_url}?{urlencode(query_params)}" if query_params else base_url
            
            if hasattr(context, 'logger') and context.logger:
                context.logger.info(f"Making request to RxNAV API for multi-ingredient brands")
            
            with urlopen(url) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}
