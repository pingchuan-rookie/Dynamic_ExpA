import tushare as ts
import os
import json
from typing import Dict, Any
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class TmtTwincomeTool(Tool):
    
    def __init__(self):
        super().__init__()
        self._pro_api = None
        self._name = "台湾电子产业月营收"
        
    
    def _get_pro_api(self):
        """Initialize Tushare Pro API client."""
        if self._pro_api is None:
            token = os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_KEY")
            if not token:
                raise ValueError(
                    "Tushare token not found. Please set TUSHARE_TOKEN or TUSHARE_API_KEY "
                    "environment variable with your Tushare Pro token."
                )
            ts.set_token(token)
            self._pro_api = ts.pro_api()
        return self._pro_api
        
    def execute(self, context: ExecutionContext, params: Dict[str, Any]) -> Any:
        try:
            pro = self._get_pro_api()
            df = pro.tmt_twincome(**params)
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            return {"error": str(e)}
