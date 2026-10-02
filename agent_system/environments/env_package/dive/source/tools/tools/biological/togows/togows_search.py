
from typing import Dict, Any
import time
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class TogoWSSearchTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio import TogoWS
            
            database = params.get('database', '')
            query = params.get('query', '')
            offset = params.get('offset', 1)
            limit = params.get('limit', 10)
            format_type = 'json'
            
            if not database:
                return {"error": "数据库参数是必需的"}
            
            if not query:
                return {"error": "查询参数是必需的"}
            
            if offset < 1:
                return {"error": "偏移量必须大于等于1"}
            
            if limit <= 0:
                return {"error": "限制数量必须大于0"}
            
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    with TogoWS.search(database, query, offset=offset, limit=limit, format=format_type) as handle:
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
