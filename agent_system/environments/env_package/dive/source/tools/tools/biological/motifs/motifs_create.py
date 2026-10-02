
from typing import Dict, Any, List
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class MotifsCreateTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from Bio import motifs
            from Bio.Seq import Seq
            
            sequences = params.get('sequences', [])
            alphabet = params.get('alphabet', 'ACGT')
            
            if not sequences:
                return {"error": "序列列表参数是必需的"}
            
            if not isinstance(sequences, list):
                return {"error": "序列必须是列表格式"}
            
            if len(sequences) < 2:
                return {"error": "至少需要2个序列才能创建motif"}
            
            seq_objects = []
            for seq_str in sequences:
                if not isinstance(seq_str, str):
                    return {"error": "所有序列必须是字符串"}
                seq_objects.append(Seq(seq_str))
            
            motif = motifs.create(seq_objects, alphabet=alphabet)
            
            return str(motif.consensus)
                
        except Exception as e:
            return {"error": str(e)}
