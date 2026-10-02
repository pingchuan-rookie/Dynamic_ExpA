
import time
import os
import json
from typing import Dict, Any
from Bio.Data import IUPACData
from tools.core.tool import Tool


class IupacDataLettersTool(Tool):

    def execute(self, context, params: Dict[str, Any]):
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                sequence_type = params.get('sequence_type', 'all').lower()
                include_ambiguous = params.get('include_ambiguous', True)
                
                result = {}
                
                if sequence_type in ['dna', 'all']:
                    dna_data = {
                        'unambiguous_letters': IUPACData.unambiguous_dna_letters,
                        'extended_letters': IUPACData.extended_dna_letters
                    }
                    if include_ambiguous:
                        dna_data['ambiguous_letters'] = IUPACData.ambiguous_dna_letters
                        dna_data['ambiguous_values'] = dict(IUPACData.ambiguous_dna_values)
                        dna_data['complement_table'] = dict(IUPACData.ambiguous_dna_complement)
                    
                    result['dna'] = dna_data
                
                if sequence_type in ['rna', 'all']:
                    rna_data = {
                        'unambiguous_letters': IUPACData.unambiguous_rna_letters
                    }
                    if include_ambiguous:
                        rna_data['ambiguous_letters'] = IUPACData.ambiguous_rna_letters
                        rna_data['ambiguous_values'] = dict(IUPACData.ambiguous_rna_values)
                        rna_data['complement_table'] = dict(IUPACData.ambiguous_rna_complement)
                    
                    result['rna'] = rna_data
                
                if sequence_type in ['protein', 'all']:
                    protein_data = {
                        'standard_letters': IUPACData.protein_letters,
                        'extended_letters': IUPACData.extended_protein_letters,
                        'one_to_three_letter': dict(IUPACData.protein_letters_1to3),
                        'three_to_one_letter': dict(IUPACData.protein_letters_3to1)
                    }
                    if include_ambiguous:
                        protein_data['extended_values'] = dict(IUPACData.extended_protein_values)
                        protein_data['extended_one_to_three'] = dict(IUPACData.protein_letters_1to3_extended)
                        protein_data['extended_three_to_one'] = dict(IUPACData.protein_letters_3to1_extended)
                    
                    result['protein'] = protein_data
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    return {"error": f"获取IUPAC字母表失败: {str(e)}"}
                time.sleep(retry_delay)
                retry_delay *= 2
        return {"error": "Max retries exceeded"}
