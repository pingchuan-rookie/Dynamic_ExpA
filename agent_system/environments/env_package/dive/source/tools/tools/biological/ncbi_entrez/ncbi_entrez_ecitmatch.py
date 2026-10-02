#!/usr/bin/env python3

import os
import time
from typing import Dict, Any
from Bio import Entrez
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class NcbiEntrezEcitmatchTool(Tool):
    
    def _setup_entrez(self):
        config_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ncbi_key.txt')
        api_key = None
        email = None
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('api_key='):
                            api_key = line.split('=', 1)[1]
                        elif line.startswith('email='):
                            email = line.split('=', 1)[1]
            except Exception:
                pass

        api_key = api_key or os.getenv("NCBI_API_KEY", "")
        email = email or os.getenv("NCBI_EMAIL", "")

        if not hasattr(Entrez, 'email') or not Entrez.email:
            Entrez.email = email if email else "user@example.org"
        
        if api_key and (not hasattr(Entrez, 'api_key') or not Entrez.api_key):
            Entrez.api_key = api_key
    
    def _validate_citation_string(self, citation: str) -> bool:
        if not citation or not isinstance(citation, str):
            return False
        
        parts = citation.split('|')
        if len(parts) < 4:
            return False
        
        try:
            year = parts[1].strip()
            if year and not year.isdigit():
                return False
        except:
            return False
        
        return True
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._setup_entrez()
            
            bdata = params.get('bdata')
            
            if not bdata:
                return {"error": "Missing required parameter: bdata"}
            
            if not isinstance(bdata, str) or len(bdata.strip()) == 0:
                return {"error": "bdata must be a non-empty string"}
            
            citations = [line.strip() for line in bdata.strip().split('\n') if line.strip()]
            for citation in citations:
                if not self._validate_citation_string(citation):
                    return {"error": f"Invalid citation format: {citation[:50]}..."}
            
            ecitmatch_params = {
                'bdata': bdata.strip(),
                'retmode': 'text'
            }
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    handle = Entrez.ecitmatch(**ecitmatch_params)
                    
                    result = handle.read()
                    if isinstance(result, bytes):
                        result = result.decode('utf-8')
                    handle.close()
                    return {"result": result}
                    
                except Exception as retry_e:
                    if attempt < max_retries - 1 and ("HTTP Error 429" in str(retry_e) or "URLError" in str(retry_e)):
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
                    
        except Exception as e:
            error_msg = str(e)
            if "HTTP Error 400" in error_msg:
                error_msg = "Invalid citation format. Check your citation strings."
            elif "URLError" in error_msg:
                error_msg = "Network error. Please check your internet connection."
            
            return {"error": f"NCBI ECitMatch failed: {error_msg}"}
