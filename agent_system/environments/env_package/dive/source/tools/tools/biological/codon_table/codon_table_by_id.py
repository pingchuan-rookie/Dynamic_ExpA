
import time
import os
import json
from typing import Dict, Any
from Bio.Data import CodonTable
from tools.core.tool import Tool


class CodonTableByIdTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                table_id = params.get('table_id')
                include_forward_table = params.get('include_forward_table', True)
                include_back_table = params.get('include_back_table', False)
                
                if table_id is None:
                    return {"error": "遗传密码表ID是必需的"}
                
                all_tables = CodonTable.ambiguous_dna_by_id
                if table_id not in all_tables:
                    available_ids = list(all_tables.keys())
                    return {"error": f"未找到ID为{table_id}的遗传密码表。可用ID: {available_ids}"}
                
                table = all_tables[table_id]
                
                result = {
                    'id': table.id,
                    'names': list(table.names),
                    'start_codons': list(table.start_codons),
                    'stop_codons': list(table.stop_codons),
                    'nucleotide_alphabet': str(table.nucleotide_alphabet),
                    'protein_alphabet': str(table.protein_alphabet)
                }
                
                if include_forward_table:
                    result['forward_table'] = dict(table.forward_table)
                
                if include_back_table:
                    result['back_table'] = dict(table.back_table)
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"获取遗传密码表失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
