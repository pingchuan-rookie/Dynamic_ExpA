
from typing import Dict, Any
import time
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class TogoWSConvertTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio import TogoWS
            
            data = params.get('data', '')
            input_format = params.get('input_format', '')
            output_format = params.get('output_format', '')
            
            if not data:
                return {"error": "数据参数是必需的"}
            
            if not input_format:
                return {"error": "输入格式参数是必需的"}
            
            if not output_format:
                return {"error": "输出格式参数是必需的"}
            
            if input_format == output_format:
                return {"error": "输入格式和输出格式不能相同"}
            
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    with TogoWS.convert(data, input_format, output_format) as handle:
                        return handle.read()
                    
                except Exception as retry_e:
                    error_msg = str(retry_e)
                    
                    if "Unsupported conversion" in error_msg:
                        return {"error": error_msg}
                    
                    if attempt < max_retries - 1:
                        if "timeout" in error_msg.lower() or "URLError" in error_msg:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
                
        except Exception as e:
            return {"error": str(e)}
