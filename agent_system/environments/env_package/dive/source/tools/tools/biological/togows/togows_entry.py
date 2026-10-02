
from typing import Dict, Any
import time
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class TogoWSEntryTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio import TogoWS
            
            database = params.get('database', '')
            entry_id = params.get('entry_id', '')
            format_type = params.get('format', 'fasta')
            field = params.get('field')
            
            if not database:
                return {"error": "数据库参数是必需的"}
            
            if not entry_id:
                return {"error": "条目ID参数是必需的"}
            
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    with TogoWS.entry(database, entry_id, format=format_type, field=field) as handle:
                        return handle.read()
                    
                except Exception as retry_e:
                    if attempt < max_retries - 1:
                        if "timeout" in str(retry_e).lower() or "URLError" in str(retry_e):
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
                
        except Exception as e:
            return {"error": str(e)}
