
import time
import os
import json
from typing import Dict, Any, List
from Bio import SeqFeature
from Bio.Seq import Seq
from tools.core.tool import Tool


class SeqFeatureCompoundTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                sequence = params.get('sequence', '')
                locations = params.get('locations', [])
                extract = params.get('extract', True)
                
                if not sequence:
                    return {"error": "序列参数是必需的"}
                if not locations:
                    return {"error": "位置列表是必需的"}
                
                feature_locations = []
                for loc_data in locations:
                    start = loc_data.get('start')
                    end = loc_data.get('end')
                    strand = loc_data.get('strand')
                    
                    if start is None or end is None:
                        return {"error": f"位置数据缺少start或end: {loc_data}"}
                    
                    feature_locations.append(
                        SeqFeature.FeatureLocation(start, end, strand)
                    )
                
                compound_location = SeqFeature.CompoundLocation(feature_locations)
                
                result = {
                    'part_count': len(compound_location.parts),
                    'total_length': len(compound_location),
                    'sequence_length': len(sequence),
                    'parts': []
                }
                
                for i, part in enumerate(compound_location.parts):
                    part_info = {
                        'index': i,
                        'start': part.start,
                        'end': part.end,
                        'strand': part.strand,
                        'length': len(part)
                    }
                    result['parts'].append(part_info)
                
                if extract:
                    seq_obj = Seq(sequence)
                    extracted_seq = compound_location.extract(seq_obj)
                    result['extracted_sequence'] = str(extracted_seq)
                    result['extracted_length'] = len(extracted_seq)
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"复合序列特征位置处理失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
