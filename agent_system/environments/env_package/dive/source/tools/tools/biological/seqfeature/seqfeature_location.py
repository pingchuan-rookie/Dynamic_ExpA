
import time
import os
import json
from typing import Dict, Any
from Bio import SeqFeature
from Bio.Seq import Seq
from tools.core.tool import Tool


class SeqFeatureLocationTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                sequence = params.get('sequence', '')
                start = params.get('start')
                end = params.get('end')
                strand = params.get('strand')
                extract = params.get('extract', True)
                
                if not sequence:
                    return {"error": "序列参数是必需的"}
                if start is None:
                    return {"error": "起始位置是必需的"}
                if end is None:
                    return {"error": "结束位置是必需的"}
                
                location = SeqFeature.FeatureLocation(start, end, strand)
                
                result = {
                    'start': location.start,
                    'end': location.end,
                    'strand': location.strand,
                    'length': len(location),
                    'sequence_length': len(sequence)
                }
                
                if extract:
                    seq_obj = Seq(sequence)
                    extracted_seq = location.extract(seq_obj)
                    result['extracted_sequence'] = str(extracted_seq)
                    result['extracted_length'] = len(extracted_seq)
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"序列特征位置处理失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
