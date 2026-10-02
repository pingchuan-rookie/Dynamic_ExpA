#!/usr/bin/env python3

from typing import Dict, Any
from Bio.Seq import transcribe, back_transcribe
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class BioSeqTranscribeTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sequence = params.get('sequence')
            operation = params.get('operation', 'transcribe')
            
            if not sequence:
                return {"error": "Missing required parameter: sequence"}
            
            clean_sequence = ''.join(sequence.upper().split())
            
            if operation == 'transcribe':
                result = str(transcribe(clean_sequence))
            elif operation == 'back_transcribe':
                result = str(back_transcribe(clean_sequence))
            else:
                return {"error": f"Invalid operation: {operation}"}
            
            return {"result": result}
            
        except Exception as e:
            return {"error": f"Transcription operation failed: {str(e)}"}
