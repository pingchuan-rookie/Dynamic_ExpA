
import time
import os
import json
from typing import Dict, Any
from Bio import Cluster
from tools.core.tool import Tool


class ClusterPcaTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                data = params.get('data')
                
                columnmean, coordinates, components, eigenvalues = Cluster.pca(data)
                
                return {
                    'columnmean': columnmean.tolist() if hasattr(columnmean, 'tolist') else list(columnmean),
                    'coordinates': coordinates.tolist() if hasattr(coordinates, 'tolist') else [row.tolist() if hasattr(row, 'tolist') else list(row) for row in coordinates],
                    'components': components.tolist() if hasattr(components, 'tolist') else [row.tolist() if hasattr(row, 'tolist') else list(row) for row in components],
                    'eigenvalues': eigenvalues.tolist() if hasattr(eigenvalues, 'tolist') else list(eigenvalues)
                }
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"PCA analysis failed: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
