
import time
import os
import json
from typing import Dict, Any
from Bio import Cluster
from tools.core.tool import Tool


class ClusterTreeclusterTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                data = params.get('data')
                method = params.get('method', 'm')
                dist = params.get('dist', 'e')
                transpose = params.get('transpose', False)
                
                tree = Cluster.treecluster(
                    data=data,
                    method=method,
                    dist=dist,
                    transpose=transpose
                )
                
                tree_data = []
                for i in range(len(tree)):
                    node = tree[i]
                    tree_data.append({
                        'left': node.left,
                        'right': node.right, 
                        'distance': node.distance
                    })
                
                return tree_data
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"Hierarchical clustering failed: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
