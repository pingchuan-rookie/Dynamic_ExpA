
import json
import os
import time
from typing import Any, Dict

from Bio import Entrez

from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class NcbiEntrezSearchTool(Tool):
    
    def _setup_entrez(self):
        config_file = 'ncbi_key.txt'
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
            
            db = params.get('db')
            term = params.get('term')
            
            if not db:
                return {"error": "Missing required parameter: db"}
            
            if not self._validate_database(db):
                return {"error": f"Invalid database: {db}"}
            
            if not term:
                return {"error": "Missing required parameter: term"}
            
            search_params = {
                'db': db,
                'term': term
            }
            
            if 'retmax' in params:
                retmax = params['retmax']
                if not isinstance(retmax, int) or retmax < 1 or retmax > 100000:
                    return {"error": "Parameter 'retmax' must be an integer between 1 and 100000"}
                search_params['retmax'] = retmax
            
            if 'retstart' in params:
                retstart = params['retstart']
                if not isinstance(retstart, int) or retstart < 0:
                    return {"error": "Parameter 'retstart' must be a non-negative integer"}
                search_params['retstart'] = retstart
            
            for param in ['sort', 'rettype', 'datetype', 'mindate', 'maxdate']:
                if param in params:
                    search_params[param] = params[param]
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    handle = Entrez.esearch(**search_params)
                    record = Entrez.read(handle)
                    handle.close()
                    return record
                except Exception as retry_e:
                    if attempt < max_retries - 1:
                        if "HTTP Error 429" in str(retry_e) or "URLError" in str(retry_e):
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
            
        except Exception as e:
            error_msg = str(e)
            if "HTTP Error 400" in error_msg:
                error_msg = "Invalid query syntax. Check your search terms."
            elif "URLError" in error_msg:
                error_msg = "Network error. Please check your internet connection."
            return {"error": f"NCBI ESearch failed: {error_msg}"}
