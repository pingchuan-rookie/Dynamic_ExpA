
import time
import json
from typing import Dict, Any
from tools.core.tool import Tool
from tools.core.types import ExecutionContext
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ProtParamIsoelectricPointTool(Tool):

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Any:
        try:
            sequence = params.get('sequence')
            
            if not sequence:
                return {"error": "缺少必需参数: sequence"}
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    analysis = ProteinAnalysis(sequence)
                    return analysis.isoelectric_point()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1 * (attempt + 1))
            return {"error": "Max retries exceeded"}
                    
        except Exception as e:
            return {"error": f"等电点计算失败: {str(e)}"}

    @classmethod
