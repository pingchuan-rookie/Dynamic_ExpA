#!/usr/bin/env python3

import os
import time
from typing import Dict, Any
from Bio import Entrez
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class NcbiEntrezEspellTool(Tool):
    
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
    
    def _validate_database(self, db: str) -> bool:
        valid_dbs = {
            'pubmed', 'protein', 'nucleotide', 'nuccore', 'gene', 'genome', 
            'structure', 'pmc', 'taxonomy', 'snp', 'geo', 'sra', 'books',
            'cancerchromosomes', 'cdd', 'gap', 'domains', 'genomeprj', 
            'gensat', 'gds', 'homologene', 'journals', 'mesh', 'ncbisearch',
            'nlmcatalog', 'omia', 'omim', 'popset', 'probe', 'proteinclusters',
            'pcassay', 'pccompound', 'pcsubstance', 'toolkit', 'unigene', 'unists'
        }
        return db.lower() in valid_dbs
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._setup_entrez()
            
            term = params.get('term')
            db = params.get('db', 'pubmed')
            
            if not term:
                return {"error": "Missing required parameter: term"}
            
            if not isinstance(term, str) or len(term.strip()) == 0:
                return {"error": "Term must be a non-empty string"}
            
            if not self._validate_database(db):
                return {"error": f"Invalid database: {db}"}
            
            espell_params = {
                'term': term.strip(),
                'db': db
            }
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    handle = Entrez.espell(**espell_params)
                    record = Entrez.read(handle)
                    handle.close()
                    return record
                    
                except Exception as retry_e:
                    if attempt < max_retries - 1 and ("HTTP Error 429" in str(retry_e) or "URLError" in str(retry_e)):
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
                    
        except Exception as e:
            error_msg = str(e)
            if "HTTP Error 400" in error_msg:
                error_msg = "Invalid search term or database. Check your inputs."
            elif "URLError" in error_msg:
                error_msg = "Network error. Please check your internet connection."
            
            return {"error": f"NCBI ESpell failed: {error_msg}"}
