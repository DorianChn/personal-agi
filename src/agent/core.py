"""Agent 核心：对话管理 + 工具调用 + 记忆 + RAG（骨架）。

本模块提供可扩展的智能体编排能力，默认只启用对话管理，
工具调用 / RAG 按需在 config 中开启。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src.inference.client import InferenceClient


@dataclass
class Tool:
    """工具定义：名称、描述（供模型判断）、参数 schema、执行函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable


@dataclass
class Agent:
    """轻量智能体，封装对话与工具调用。"""

    client: InferenceClient
    system_prompt: str = "你是我的个人 AI 助手。"
    tools: dict[str, Tool] = field(default_factory=dict)
    memory: list[dict[str, str]] = field(default_factory=list)

    def chat(self, user_input: str) -> str:
        """单轮对话，带记忆与工具调用。"""
        self.memory.append({"role": "user", "content": user_input})

        # 若有工具，先让模型决定是否调用
        if self.tools:
            tool_calls = self._plan_tool_calls(user_input)
            if tool_calls:
                results = self._execute_tools(tool_calls)
                user_input = f"{user_input}\n\n[工具结果]\n{results}"

        messages = [{"role": "system", "content": self.system_prompt}, *self.memory]
        resp = self.client.chat(messages)
        self.memory.append({"role": "assistant", "content": resp})
        return resp

    def _plan_tool_calls(self, user_input: str) -> list[tuple[str, dict]]:
        """简化的工具规划：后续可替换为 function calling 正式协议。"""
        return []

    def _execute_tools(self, calls: list[tuple[str, dict]]) -> str:
        outs = []
        for name, args in calls:
            if name in self.tools:
                try:
                    outs.append(f"{name}: {self.tools[name].func(**args)}")
                except Exception as e:  # noqa: BLE001
                    outs.append(f"{name} 执行失败：{e}")
        return json.dumps(outs, ensure_ascii=False)
