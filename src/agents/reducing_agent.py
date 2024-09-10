import json
from typing import List, Tuple, Any, Optional, Union

from src.agents.tool_call_result import ToolCallResult
from src.agents.utility import validate_expression, reduce_expression

from src.tools.calculator import calculate
from src.llm.llm_base import LLMClientBase
from src.llm.chatgpt import MessageHistory


class ReducingCalculatorAgent:
    def __init__(self, llm_client: LLMClientBase, config: dict) -> None:
        self.llm_client = llm_client

        self.max_expression_length: int = config['max_expression_length']

        self.system_prompt: str = config['system_prompt']
        self.prompt: str = config['prompt']
        self.max_llm_calls: int = config['max_llm_calls']

    def run(self, expression: str) -> Optional[float]:
        print(f"Input expression: {expression}")

        validate_expression(expression, self.max_expression_length)

        steps: List[str] = []
        final_result: Optional[float] = None
        i = 1

        while True:
            # print(f'\n ================ iteration {i} ================ \n')

            prompt_msg = self._prepare_next_prompt(expression)

            # print('\n----- prompt_msg -----\n')
            # print(prompt_msg)

            response = self.llm_client.run_prompt(prompt_msg)

            result = self._process_tool_calls(response.tool_calls, expression)

            expression = result.remaining_expression

            print(f"Call {i}: {'    ,   '.join(result.call_steps)} --> remaining expression: {expression}")

            if result.is_final_step:
                final_result = result.results[-1][0]   # Last result --> first element in the tuple
                # print(f"Final result: {final_result}")
                break

            steps.extend(result.call_steps)

            if i >= self.max_llm_calls:
                raise RuntimeError(f'Max LLM calls reached before final result. Max calls: {self.max_llm_calls}')

            i += 1

        return final_result

    def _process_tool_calls(self, tool_calls: List[Any], expression: str) -> ToolCallResult:
        if not tool_calls:
            raise RuntimeError("Error: Expected a tool call but received none.")

        function_call_result_message = []

        results: List[Tuple[float, str]] = []
        is_final_step = False
        call_steps: List[str] = []

        for tool_call in tool_calls:
            function_call = tool_call.function

            try:
                func_args = json.loads(function_call.arguments)
                a = func_args['a']
                b = func_args['b']
                op = func_args['op']
                is_final_step = func_args['is_final_step']
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Invalid tool call arguments format: {function_call.arguments}. \n error: {e}")

            result = calculate(a, b, op)

            expression = reduce_expression(expression, a, b, op, result)

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


