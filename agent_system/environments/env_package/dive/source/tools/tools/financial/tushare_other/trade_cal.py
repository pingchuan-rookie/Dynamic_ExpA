import os
import json
import tushare as ts
from typing import Dict, Any
from tools.core.tool import Tool
from tools.core.types import ExecutionContext


class TradeCalTool(Tool):
    
    def __init__(self):
        super().__init__()
        self._name = "各交易所交易日历"
        self._pro_api = None
        
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
        pro = self._get_pro_api()
        df = pro.trade_cal(**params)
        return df.to_dict('records') if not df.empty else []
