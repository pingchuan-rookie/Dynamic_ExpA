
import time
import os
import json
import numpy as np
from typing import Dict, Any
from Bio import SVDSuperimposer
from tools.core.tool import Tool


class SvdSuperimposeTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                coord1 = params.get('coordinates1')
                coord2 = params.get('coordinates2')
                include_transformed = params.get('include_transformed', False)
                
                if coord1 is None:
                    return {"error": "第一组坐标是必需的"}
                if coord2 is None:
                    return {"error": "第二组坐标是必需的"}
                
                coord1_array = np.array(coord1)
                coord2_array = np.array(coord2)
                
                if coord1_array.shape != coord2_array.shape:
                    return {"error": f"坐标形状不匹配: {coord1_array.shape} vs {coord2_array.shape}"}
                
                if len(coord1_array.shape) != 2 or coord1_array.shape[1] != 3:
                    return {"error": f"坐标必须是Nx3的数组，实际形状: {coord1_array.shape}"}
                
                sup = SVDSuperimposer.SVDSuperimposer()
                
                sup.set(coord1_array, coord2_array)
                
                initial_rms = sup.get_init_rms()
                
                sup.run()
                
                final_rms = sup.get_rms()
                rot_matrix, translation = sup.get_rotran()
                
                result = {
                    'initial_rms': float(initial_rms),
                    'final_rms': float(final_rms),
                    'rotation_matrix': rot_matrix.tolist(),
                    'translation_vector': translation.tolist(),
                    'coordinate_count': len(coord1_array),
                    'improvement': float(initial_rms - final_rms)
                }
                
                if include_transformed:
                    transformed_coords = sup.get_transformed()
                    result['transformed_coordinates'] = transformed_coords.tolist()
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"3D结构叠合失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
