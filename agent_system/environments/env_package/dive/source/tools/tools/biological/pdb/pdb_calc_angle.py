
from typing import Dict, Any, List
import time
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class PdbCalcAngleTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio.PDB import calc_angle
            from Bio.PDB.vectors import Vector
            
            v1 = params.get('v1', [])
            v2 = params.get('v2', [])
            v3 = params.get('v3', [])
            
            for i, v in enumerate([v1, v2, v3], 1):
                if not isinstance(v, list) or len(v) != 3:
                    return {"error": f"v{i}必须是包含3个数值的列表"}
                for coord in v:
                    if not isinstance(coord, (int, float)):
                        return {"error": f"v{i}的坐标必须是数值"}
            
            max_retries = 2
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    vec1 = Vector(*v1)
                    vec2 = Vector(*v2)
                    vec3 = Vector(*v3)
                    
                    angle = calc_angle(vec1, vec2, vec3)
                    
                    return float(angle)
                    
                except Exception as retry_e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise retry_e
            return {"error": "Max retries exceeded"}
                
        except Exception as e:
            return {"error": str(e)}
