
import time
import os
import json
from typing import Dict, Any
from Bio.Seq import back_transcribe
from tools.core.tool import Tool


class BioSeqBackTranscribeTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                sequence = params.get('sequence', '')
                
                if not sequence:
                    return {"error": "序列参数是必需的"}
                
                dna_sequence = back_transcribe(sequence)
                
                return str(dna_sequence)
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"RNA反转录失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
