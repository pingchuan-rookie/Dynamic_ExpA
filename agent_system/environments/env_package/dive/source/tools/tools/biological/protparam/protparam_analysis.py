
import time
import json
from typing import Dict, Any
from tools.core.tool import Tool
from tools.core.types import ExecutionContext
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ProtParamAnalysisTool(Tool):

    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Any:
        try:
            sequence = params.get('sequence')
            
            if not sequence:
                return {"error": "缺少必需参数: sequence"}
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    analysis = ProteinAnalysis(sequence)
                    
                    result = {
                        'molecular_weight': analysis.molecular_weight(),
                        'isoelectric_point': analysis.isoelectric_point(),
                        'aromaticity': analysis.aromaticity(),
                        'instability_index': analysis.instability_index(),
                        'gravy': analysis.gravy(),
                        'amino_acids_percent': analysis.amino_acids_percent,
                        'amino_acids_count': analysis.count_amino_acids(),
                        'length': analysis.length,
                        'secondary_structure_fraction': analysis.secondary_structure_fraction(),
                        'flexibility': analysis.flexibility(),
                        'molar_extinction_coefficient': analysis.molar_extinction_coefficient(),
                        'monoisotopic': analysis.monoisotopic
                    }
                    
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1 * (attempt + 1))
            return {"error": "Max retries exceeded"}
                    
        except Exception as e:
            return {"error": f"蛋白质分析失败: {str(e)}"}

    @classmethod
