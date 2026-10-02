
import time
import os
import json
from typing import Dict, Any
from Bio import Restriction
from tools.core.tool import Tool


class RestrictionAllEnzymesTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                limit = params.get('limit', 100)
                search_pattern = params.get('search_pattern', '')
                
                all_enzymes = Restriction.AllEnzymes
                total_count = len(all_enzymes)
                
                enzyme_names = [str(enzyme) for enzyme in all_enzymes]
                
                if search_pattern:
                    enzyme_names = [name for name in enzyme_names 
                                   if search_pattern.lower() in name.lower()]
                
                limited_enzymes = enzyme_names[:limit]
                
                return {
                    'total_enzymes': total_count,
                    'filtered_count': len(enzyme_names) if search_pattern else total_count,
                    'returned_count': len(limited_enzymes),
                    'search_pattern': search_pattern if search_pattern else None,
                    'limit': limit,
                    'enzymes': limited_enzymes
                }
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"酶列表获取失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
