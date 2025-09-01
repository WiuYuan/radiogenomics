# src/services/custom_tools.py

import glob
from typing import Callable, Dict, List
from src.services.llm import LLM
import re
import os
import subprocess

MAIN_DIR = "/Users/yuanwen/Desktop/Docker_Environment/intern2/2"
PYTHON_PATH = "/Users/yuanwen/anaconda3/envs/env_early/bin/python"
MATLAB_PATH = "/Applications/MATLAB_R2023b.app/bin/matlab"


class custom_tools:
    def __init__(
        self,
        MAIN_DIR="/Users/yuanwen/Desktop/Docker_Environment/intern2/2",
        PYTHON_PATH="/Users/yuanwen/anaconda3/envs/env_early/bin/python",
        MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
        llm=LLM(),
    ):
        self.MAIN_DIR = MAIN_DIR
        self.PYTHON_PATH = PYTHON_PATH
        self.MATLAB_PATH = MATLAB_PATH
        self.llm = llm

    def func_error(self, error: str):
        prompt_file = "src/services/func_error_prompt.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(error_message=error, MAIN_DIR=self.MAIN_DIR)
        return self.llm.query(prompt)

    def func_cmd(self, command: str):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        error = result.stderr.strip()
        if result.returncode == 0:
            return output
        else:
            error = self.func_error(error)
            return f"Error (code {result.returncode}):\nSTDOUT:\n{output}\nSTDERR:\n{error}"

    def func_ls(self, filepath: str):
        """
        View the contents of a file at a specified path.
        Example: input 'agent1/main.sh' will show the contents of '$MAIN_DIR/agent1/main.sh'
        """
        base_path = MAIN_DIR
        full_path = os.path.join(base_path, filepath)
        command = f"cd {full_path} && ls --color=never"
        return self.func_cmd(command)

    def func_cat(self, filepath: str):
        base_path = MAIN_DIR
        full_path = os.path.join(base_path, filepath)
        command = f"cat {full_path}"
        return self.func_cmd(command)

    def func_write(self, filepath: str, text: str):
        """
        Write text to a file at a specified filepath.
        """
        # text = text.replace("\\n", "\n")
        base_path = MAIN_DIR
        full_path = os.path.join(base_path, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"Written to {filepath}"

    def func_write_a(self, input: str):
        parts = input.split("|||", 1)
        if len(parts) != 2:
            return "Error: input format must be 'filepath|||text'"

        filepath, text = parts
        base_path = MAIN_DIR
        full_path = os.path.join(base_path, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, "a", encoding="utf-8") as f:
                f.write(text)
            return f"Appended to {filepath}"
        except Exception as e:
            return f"Exception: {e}"

    def func_python(self, filepath: str):
        """
        Run a Python script at the specified path under ${MAIN_DIR}.
        The agent will first 'cd ${MAIN_DIR}' and then execute the script.
        Example: input 'agent1/test/test.py' will run
        'cd ${MAIN_DIR} && python ${MAIN_DIR}/agent1/test/test.py' using the designated Python environment and return the output.
        """
        base_path = MAIN_DIR
        full_path = os.path.join(base_path, filepath)
        command = f"cd {MAIN_DIR} && {PYTHON_PATH} {full_path}"
        return self.func_cmd(command)

    def func_matlab(self, filepath: str):
        """
        Run a MATLAB script at the specified path under ${MAIN_DIR}.
        The agent will first 'cd ${MAIN_DIR}' and then execute the script in MATLAB -batch mode.
        Example: input 'agent1/test/test.m' will run '${MAIN_DIR}/agent1/test/test.m' and return the output.
        """
        base_path = "/Users/yuanwen/Desktop/Docker_Environment/intern2/2"
        full_path = os.path.join(base_path, filepath)
        dir_path = os.path.dirname(full_path)
        script_name = os.path.splitext(os.path.basename(full_path))[0]
        command = (
            f"cd {MAIN_DIR} && {MATLAB_PATH} -batch \"cd('{dir_path}'); {script_name}\""
        )
        return self.func_cmd(command)

    def func_human(self, error: str):
        """
        Use this tool when the AI cannot resolve an issue automatically.
        It prompts a human to provide a solution to the given error or problem, and returns the human-provided response.
        """
        print(f"\nError occurred:\n{error}\n")
        user_input = input("Please provide your solution: ")
        return user_input

    def func_view(self, filepath: str, request):
        """
        Let AI view and analyze the content of a specified file.
        - filepath: the path to the file to read
        - request: a question or instruction regarding the file content
        The tool will read the file content and, together with the provided instruction, return an AI analysis or answer based on the file.
        """
        prompt_file = "src/services/func_view_prompt.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        file = self.func_cat(filepath)
        prompt = prompt_template.format(request=request, file=file)
        return self.llm.query(prompt)

    def func_generate_code(self, filepath: str, request: str):
        """
        Generate complete runnable code based strictly on the request and write it into the specified file.
        - filepath: full path to the file to create or update
        - request: the code specification, including all necessary information such as data locations, file paths, dependencies, and expected behavior
        Important:
        - The AI does NOT have any prior knowledge about your environment or files. Everything needed to generate correct code must be included in the request.
        - Output must be fully functional code without explanations, comments, or extra text.
        - If the request cannot be fulfilled, output 'NOT_FOUND'.
        """
        prompt_filepath = "src/services/func_generate_code_prompt.txt"
        prompt_template = self.func_cat(prompt_filepath)
        prompt = prompt_template.format(request=request)
        code = self.llm.query(prompt)
        return f"Code generate completedly. {self.func_write(filepath, code)}"
