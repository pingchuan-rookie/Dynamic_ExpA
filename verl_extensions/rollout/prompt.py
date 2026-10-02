# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run the complete processor operation off the asyncio event loop.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


async def process_multimodal_prompt(self, messages, tools, images, videos, audios, mm_processor_kwargs):
    # Render native text protocols and preserve their actual decision boundary.
    from verl.experimental.agent_loop.agent_loop import (
        _PROCESSOR_EXECUTOR,
        apply_chat_template,
        build_multimodal_processor_inputs,
        normalize_token_ids,
    )

    def process_prompt():
        raw_prompt = apply_chat_template(
            self.processor,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[raw_prompt],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=mm_processor_kwargs
            if mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        return normalize_token_ids(model_inputs.pop("input_ids"))

    prompt_ids = await self.loop.run_in_executor(_PROCESSOR_EXECUTOR, process_prompt)
    return prompt_ids
