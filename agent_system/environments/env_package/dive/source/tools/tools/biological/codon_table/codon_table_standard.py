
import time
import os
import json
from typing import Dict, Any
from Bio.Data import CodonTable
from tools.core.tool import Tool


class CodonTableStandardTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                include_forward_table = params.get('include_forward_table', True)
                include_back_table = params.get('include_back_table', False)
                
                standard_table = CodonTable.standard_dna_table
                
                result = {
                    'id': standard_table.id,
                    'names': list(standard_table.names),
                    'start_codons': list(standard_table.start_codons),
                    'stop_codons': list(standard_table.stop_codons),
                    'nucleotide_alphabet': str(standard_table.nucleotide_alphabet),
                    'protein_alphabet': str(standard_table.protein_alphabet)
                }
                
                if include_forward_table:
                    result['forward_table'] = dict(standard_table.forward_table)
                
                if include_back_table:
                    result['back_table'] = dict(standard_table.back_table)
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"获取标准遗传密码表失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
