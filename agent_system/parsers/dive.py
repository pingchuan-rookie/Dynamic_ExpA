"""DIVE compatibility names for the shared native tool-call parser."""
from agent_system.parsers.native_tools import (
    NativeToolFormatError as DiveFormatError,
    decode_response,
    native_protocol,
    parse_response,
)
from verl.experimental.agent_loop.tool_parser import ToolParser


@ToolParser.register("dive")
class DiveToolParser(ToolParser):
    async def extract_tool_calls(self, responses_ids, tools=None):
        return self.tokenizer.decode(responses_ids, skip_special_tokens=False), []
