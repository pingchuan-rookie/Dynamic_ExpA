
from typing import Dict, Any
import time
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class PdbIsNucleicTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio.PDB import is_nucleic
            
            residue = params.get('residue', '')
            standard = params.get('standard', False)
            
            if not residue:
                return {"error": "残基参数是必需的"}
            
            max_retries = 2
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    result = is_nucleic(residue, standard=standard)
                    
                    return result
                    
                except Exception as retry_e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
                
        except Exception as e:
            return {"error": str(e)}
