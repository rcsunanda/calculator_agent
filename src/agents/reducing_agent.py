import json
import re
from typing import List, Tuple, Any, Optional, Union

from src.agents.tool_call_result import ToolCallResult
from src.tools.calculator import calculate
from src.llm.llm_base import LLMClientBase
from src.llm.chatgpt import MessageHistory


def float_to_str(f: float) -> str:
    return f'{f: g}'


def create_number_pattern(num: Union[int, float]) -> str:
    if isinstance(num, int):
        return r'\b' + str(num) + r'\b'
    else:
        # Handle both integer and decimal representations
        return r'\b' + float_to_str(num).replace('.', r'\.?') + r'\b'


class ReducingCalculatorAgent:
    def __init__(self, llm_client: LLMClientBase, config: dict) -> None:
        self.llm_client = llm_client

        self.system_prompt: str = config['system_prompt']
        self.prompt: str = config['prompt']
        self.max_calls: int = config['max_llm_calls']

    def run(self, expression: str) -> Optional[float]:
        print(f"Input expression: {expression}")

        steps: List[str] = []
        final_result: Optional[float] = None
        i = 1

        while True:
            print(f'\n ================ iteration {i} ================ \n')

            prompt_msg = self._prepare_next_prompt(expression)

            # print('\n----- prompt_msg -----\n')
            # print(prompt_msg)

            response = self.llm_client.run_prompt(prompt_msg)

            result = self._process_tool_calls(response.tool_calls, expression)

            expression = result.remaining_expression

            print(f"Call {i}: {'    ,   '.join(result.call_steps)} --> remaining expression: {expression}")

            if result.is_final_step:
                final_result = result.results[-1][0]   # Last result --> first element in the tuple
                print(f"Final result: {final_result}")
                break

            steps.extend(result.call_steps)

            if i >= self.max_calls:
                print(f"Max calls reached: {self.max_calls}")
                break

            i += 1

        return final_result

    def _validate_expression(self, expression: str) -> bool:
        return True

    def _process_tool_calls(self, tool_calls: List[Any], expression: str) -> ToolCallResult:
        if not tool_calls:
            raise ValueError("Error: Expected a function call but received none.")

        function_call_result_message = []

        results: List[Tuple[float, str]] = []
        is_final_step = False
        call_steps: List[str] = []

        for tool_call in tool_calls:
            function_call = tool_call.function
            func_args = json.loads(function_call.arguments)

            a = func_args['a']
            b = func_args['b']
            op = func_args['op']
            is_final_step = func_args['is_final_step']

            result = calculate(a, b, op)

            pattern = create_number_pattern(a) + r'\s*' + re.escape(op) + r'\s*' + create_number_pattern(b)
            expression = re.sub(pattern, float_to_str(result), expression, count=1)

            # If expression not found before final step --> Error

            step = f"{a} {op} {b} = {result}"
            call_steps.append(step)

            results.append((result, tool_call.id))

            function_call_result_message.append({
                "role": "tool",
                "content": json.dumps({"result": result}),
                "tool_call_id": tool_call.id
            })

        return ToolCallResult(results, is_final_step, call_steps, expression)

    def _prepare_next_prompt(self, expression: str) -> MessageHistory:
        prompt = self.prompt.replace('{EXPRESSION}', expression)

        prompt_msg = MessageHistory()
        prompt_msg.add_system_message(self.system_prompt)
        prompt_msg.add_user_message(prompt)

        return prompt_msg


