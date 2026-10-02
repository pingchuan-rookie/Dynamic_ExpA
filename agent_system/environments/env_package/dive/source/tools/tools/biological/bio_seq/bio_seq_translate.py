#!/usr/bin/env python3

from typing import Dict, Any
from Bio.Seq import translate
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class BioSeqTranslateTool(Tool):
    
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sequence = params.get('sequence')
            
            if not sequence:
                return {"error": "Missing required parameter: sequence"}
            
            clean_sequence = ''.join(sequence.upper().split())
            
            translate_params = {'sequence': clean_sequence}
            
            if 'table' in params:
                translate_params['table'] = params['table']
            if 'stop_symbol' in params:
                translate_params['stop_symbol'] = params['stop_symbol']
            if 'to_stop' in params:
                translate_params['to_stop'] = params['to_stop']
            if 'cds' in params:
                translate_params['cds'] = params['cds']
            if 'gap' in params:
                translate_params['gap'] = params['gap']
            
            protein_sequence = translate(**translate_params)
            
            return {"protein_sequence": str(protein_sequence)}
            
        except Exception as e:
            return {"error": f"Translation failed: {str(e)}"}
