
import time
import os
import json
from typing import Dict, Any
from Bio.Data import CodonTable
from tools.core.tool import Tool


class CodonTableListTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                include_details = params.get('include_details', True)
                
                all_tables = CodonTable.ambiguous_dna_by_id
                
                result = {
                    'total_tables': len(all_tables),
                    'available_ids': sorted(list(all_tables.keys())),
                    'tables': []
                }
                
                for table_id in sorted(all_tables.keys()):
                    table = all_tables[table_id]
                    
                    table_info = {
                        'id': table.id,
                        'names': list(table.names),
                        'primary_name': table.names[0] if table.names else f"Table {table_id}"
                    }
                    
                    if include_details:
                        table_info.update({
                            'start_codons_count': len(table.start_codons),
                            'stop_codons_count': len(table.stop_codons),
                            'start_codons': list(table.start_codons),
                            'stop_codons': list(table.stop_codons)
                        })
                    
                    result['tables'].append(table_info)
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"获取遗传密码表列表失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
