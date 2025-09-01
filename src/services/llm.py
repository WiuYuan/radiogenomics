# src/services/llm.py

import requests
import json
import inspect
from typing import Callable, List, Dict, Any
import os
from collections import defaultdict
import json

os.environ["NO_PROXY"] = "*"


class LLM:
    def __init__(
        self,
        api_key: str = "",
        llm_url: str = "http://localhost:11434/api/chat",
        model_name: str = "qwen3:8b",
        remove_think: str = True,
        proxies: dict = None,
        format: str = "ollama",
    ):
        """
        Initialize the LLM instance.

        Parameters:
        llm_url (str): The URL of the LLM service (e.g., Ollama API endpoint).
        model_name (str): The model to use. Default is "qwen3:32b".
        remove_think (bool): Whether to remove <think>...</think> sections from the response. Default is True.
        """
        self.api_key = api_key
        self.llm_url = llm_url
        self.model_name = model_name
        self.remove_think_enabled = remove_think
        self.proxies = proxies or {"http": None, "https": None}
        self.format = format

    def remove_think(self, text: str) -> str:
        """
        Remove <think>...</think> sections from the text and trim surrounding whitespace.

        Parameters:
        text (str): The input text containing potential <think> sections.

        Returns:
        str: Cleaned text without <think> blocks.
        """
        start_tag = "<think>"
        end_tag = "</think>"

        start_idx = text.find(start_tag)
        if start_idx != -1:
            end_idx = text.find(end_tag, start_idx)
            if end_idx != -1:
                # Remove the entire <think> block including the tags
                text = text[:start_idx] + text[end_idx + len(end_tag) :]

        # Trim whitespace at the start and end
        return text.strip()

    def query(self, prompt: str, verbose: bool = True) -> str:
        if verbose:
            print(prompt)
        messages = [{"role": "user", "content": prompt}]
        return self.query_messages(messages, verbose=verbose)

    def query_with_tools(
        self,
        prompt: str,
        max_steps: int,
        tools=None,
        verbose: bool = True,
    ) -> str:
        if verbose:
            print(prompt)
        func_dict = {func.__name__: func for func in tools}
        messages = [{"role": "user", "content": prompt}]
        for _ in range(max_steps):
            print(messages)
            text, tool_calls = self.query_messages_with_tools(
                messages, tools=tools, verbose=verbose
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }
            )
            tool_results = []
            for call in tool_calls:
                if self.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["arguments"]

                if func_name in func_dict:
                    if verbose:
                        print(
                            f"\nCalling function '{func_name}' with arguments: {args}"
                        )
                    result = func_dict[func_name](**args)
                else:
                    result = f"Function {func_name} not found"

                tool_results.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": call.get("id", ""),
                        "content": str(result),
                    }
                )
            messages.extend(tool_results)
        return text

    def query_messages(self, messages: str, verbose: bool = True) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        text_accumulate = ""

        # if self.model_name == "human":
        #     open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_prompt.txt",
        #         "w",
        #         encoding="utf-8",
        #     ).write(prompt)
        #     option = input()
        #     if option == "1":
        #         raise RuntimeError("exit")
        #     text_accumulate = open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_answer.txt",
        #         "r",
        #     ).read()
        #     return text_accumulate

        # Make a streaming POST request
        with requests.post(
            self.llm_url,
            headers=headers,
            json=payload,
            proxies=self.proxies,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                # print(line)
                line_str = line.decode("utf-8").strip()
                if self.format == "ollama":
                    chunk = json.loads(line_str)
                    token = None
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            print(token, end="", flush=True)

                if self.format == "deepseek-chat":
                    line_str = line_str[len("data: ") :]
                    if line_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Failed to parse JSON from line: {line_str}"
                        ) from e
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            print(token, end="", flush=True)

        # Optionally remove <think> blocks
        if self.remove_think_enabled:
            text_accumulate = self.remove_think(text_accumulate)

        return text_accumulate

    def query_messages_with_tools(
        self, messages: str, tools=None, verbose: bool = True
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        tools = tools or []
        tools = self.create_tools(tools)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": True,
        }
        # print(payload)
        payload = {k: v for k, v in payload.items() if v is not None}

        text_accumulate = ""

        # if self.model_name == "human":
        #     open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_prompt.txt",
        #         "w",
        #         encoding="utf-8",
        #     ).write(prompt)
        #     option = input()
        #     if option == "1":
        #         raise RuntimeError("exit")
        #     text_accumulate = open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_answer.txt",
        #         "r",
        #     ).read()
        #     return text_accumulate

        # Make a streaming POST request
        tool_calls = []
        with requests.post(
            self.llm_url,
            headers=headers,
            json=payload,
            proxies=self.proxies,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                # print(line)
                line_str = line.decode("utf-8").strip()
                if self.format == "ollama":
                    chunk = json.loads(line_str)
                    token = None
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            print(token, end="", flush=True)

                    if "message" in chunk and "tool_calls" in chunk["message"]:
                        tool_calls.extend(chunk["message"]["tool_calls"])

                if self.format == "deepseek-chat":
                    line_str = line_str[len("data: ") :]
                    if line_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Failed to parse JSON from line: {line_str}"
                        ) from e
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])

                    if token:
                        text_accumulate += token
                        if verbose:
                            print(token, end="", flush=True)

        # Optionally remove <think> blocks
        if self.remove_think_enabled:
            text_accumulate = self.remove_think(text_accumulate)

        if self.format == "deepseek-chat":
            grouped = defaultdict(list)
            for call in tool_calls:
                idx = call.get("index", 0)
                args = call.get("function", {}).get("arguments", "")
                grouped[idx].append(args)

            tool_calls_clean = []
            for idx, parts in grouped.items():
                full_args_str = "".join(parts).strip()
                try:
                    full_args = json.loads(full_args_str) if full_args_str else {}
                except json.JSONDecodeError:
                    full_args = full_args_str

                func_name = None
                fields = ["id", "type", "function"]
                extracted = {}
                for call in tool_calls:
                    if call.get("index") == idx:
                        if "name" in call.get("function", {}):
                            func_name = call["function"]["name"]
                        for field in fields:
                            if field in call:
                                extracted[field] = call.get(field)
                        break

                tool_calls_clean.append(
                    {
                        **extracted,
                        "index": idx,
                        "name": func_name,
                        "arguments": full_args,
                    }
                )
            tool_calls = tool_calls_clean

        return text_accumulate, tool_calls

    @classmethod
    def create_tools(cls, func_list: List[Callable]) -> List[Dict[str, Any]]:
        tools = []

        for func in func_list:
            sig = inspect.signature(func)
            func_name = func.__name__
            func_description = func.__doc__ or f"Function {func_name}"
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in (int, float):
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    else:
                        param_type = "string"

                param_description = f"Parameter {param_name}"

                properties[param_name] = {
                    "type": param_type,
                    "description": param_description,
                }

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_description.strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            tools.append(tool)

        return tools
