"""Qwen3.5-compatible whole-conversation masking for VERL SFT."""
from __future__ import annotations

import torch

from verl.utils.chat_template import apply_chat_template
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from tau2_evolve.traces_to_sft import _assistant_is_trainable


class Qwen35MultiTurnSFTDataset(MultiTurnSFTDataset):
    """Render once and train only tokens inside Qwen3.5 assistant turns.

    Qwen3.5 rejects isolated system/tool/assistant messages and may rewrite the
    final turn when another message is appended. VERL's default per-message
    renderer therefore cannot create a valid mask. The model's explicit chat
    boundaries let us mask the exact whole-conversation tokenization instead.
    """

    def _process_single_message(
        self, index, message, full_message, tools=None, enable_thinking=None
    ):
        if index != 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone(), {}

        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = dict(self.apply_chat_template_kwargs)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        assistant_trainable = [
            _assistant_is_trainable(message)
            for message in full_message
            if message.get("role") == "assistant"
        ]
        render_messages = [
            {key: value for key, value in message.items() if key != "_trainable"}
            for message in full_message
        ]
        inputs = apply_chat_template(
            processor,
            messages=render_messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        loss_mask = torch.zeros_like(input_ids)

        assistant_start = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        active = False
        assistant_index = 0
        offset = 0
        while offset < input_ids.numel():
            if input_ids[offset : offset + len(assistant_start)].tolist() == assistant_start:
                if assistant_index >= len(assistant_trainable):
                    raise ValueError("assistant mask count does not match rendered conversation")
                active = assistant_trainable[assistant_index]
                assistant_index += 1
                offset += len(assistant_start)
                continue
            if active:
                loss_mask[offset] = 1
                if input_ids[offset].item() == end_id:
                    active = False
            offset += 1
        if active:
            raise ValueError("unterminated Qwen3.5 assistant turn")
        if assistant_index != len(assistant_trainable):
            raise ValueError("not all assistant turns were found in rendered conversation")
        return input_ids, loss_mask, attention_mask, {}
