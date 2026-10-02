#!/usr/bin/env python3

from typing import Dict, Any
from Bio.Seq import complement, reverse_complement, complement_rna, reverse_complement_rna
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class BioSeqComplementTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sequence = params.get('sequence')
            operation = params.get('operation', 'reverse_complement')
            
            if not sequence:
                return {"error": "Missing required parameter: sequence"}
            
            clean_sequence = ''.join(sequence.upper().split())
            
            if operation == 'complement':
                result = str(complement(clean_sequence))
            elif operation == 'reverse_complement':
                result = str(reverse_complement(clean_sequence))
            elif operation == 'complement_rna':
                result = str(complement_rna(clean_sequence))
            elif operation == 'reverse_complement_rna':
                result = str(reverse_complement_rna(clean_sequence))
            else:
                return {"error": f"Invalid operation: {operation}"}
            
            return {"result": result}
            
        except Exception as e:
            return {"error": f"Complement operation failed: {str(e)}"}
