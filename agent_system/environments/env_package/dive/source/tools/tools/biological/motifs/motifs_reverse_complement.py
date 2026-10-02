
from typing import Dict, Any
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class MotifsReverseComplementTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio import motifs
            from Bio.Seq import Seq
            
            sequence = params.get('sequence', '')
            inplace = params.get('inplace', False)
            
            if not sequence:
                return {"error": "序列参数是必需的"}
            
            seq_obj = Seq(sequence)
            
            result = motifs.reverse_complement(seq_obj, inplace=inplace)
            
            return str(result)
                
        except Exception as e:
            return {"error": str(e)}
