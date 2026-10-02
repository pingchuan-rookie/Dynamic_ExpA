
import time
import os
import json
from typing import Dict, Any
from Bio import Restriction
from Bio.Seq import Seq
from tools.core.tool import Tool


class RestrictionCatalyseTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                sequence = params.get('sequence', '')
                enzyme_name = params.get('enzyme_name', '')
                
                if not sequence:
                    return {"error": "序列参数是必需的"}
                if not enzyme_name:
                    return {"error": "酶名称参数是必需的"}
                
                if not hasattr(Restriction, enzyme_name):
                    return {"error": f"未找到限制性内切酶: {enzyme_name}"}
                
                enzyme = getattr(Restriction, enzyme_name)
                
                seq_obj = Seq(sequence)
                fragments = enzyme.catalyse(seq_obj)
                
                return {
                    'enzyme_name': enzyme_name,
                    'recognition_site': str(enzyme.site),
                    'original_sequence': sequence,
                    'fragments': [str(frag) for frag in fragments],
                    'fragment_count': len(fragments),
                    'fragment_lengths': [len(frag) for frag in fragments]
                }
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"限制性内切酶切割失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
