
import time
import os
import json
from typing import Dict, Any
from Bio import Cluster
from tools.core.tool import Tool


class ClusterDistancematrixTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                data = params.get('data')
                dist = params.get('dist', 'e')
                transpose = params.get('transpose', False)
                
                distance_matrix = Cluster.distancematrix(
                    data=data,
                    dist=dist,
                    transpose=transpose
                )
                
                if hasattr(distance_matrix, 'tolist'):
                    return distance_matrix.tolist()
                else:
                    return [row.tolist() if hasattr(row, 'tolist') else list(row) 
                           for row in distance_matrix]
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"Distance matrix calculation failed: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
