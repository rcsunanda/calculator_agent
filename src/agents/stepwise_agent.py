import json
from typing import List, Tuple, Any, Optional

from src.agents.tool_call_result import ToolCallResult
from src.agents.utility import validate_expression

from src.tools.calculator import calculate
from src.llm.llm_base import LLMClientBase
from src.llm.chatgpt import MessageHistory


class StepwiseCalculatorAgent:
    def __init__(self, llm_client: LLMClientBase, config: dict) -> None:
        self.llm_client = llm_client

        self.max_expression_length: int = config['max_expression_length']

        self.system_prompt: str = config['system_prompt']
        self.subsequent_prompt: str = config['subsequent_prompt']
        self.initial_prompt: str = config['initial_prompt']

        self.max_llm_calls: int = config['max_llm_calls']
        self.return_tool_call_msgs: bool = config['return_tool_call_msgs']
        self.append_messages: bool = config['append_messages']

    def run(self, expression: str) -> Optional[float]:
        print(f"Input expression: {expression}")

        validate_expression(expression, self.max_expression_length)

        initial_prompt = self.initial_prompt.replace('{EXPRESSION}', expression)

        prompt_msg = MessageHistory()
        prompt_msg.add_system_message(self.system_prompt)
        prompt_msg.add_user_message(initial_prompt)

        steps: List[str] = []
        final_result: Optional[float] = None
        i = 1

        while True:
            # print(f'\n ================ iteration {i} ================ \n')

            # print('\n----- prompt_msg -----\n')
            # print(prompt_msg)

            response = self.llm_client.run_prompt(prompt_msg)

            result = self._process_tool_calls(response.tool_calls)

            print(f"Call {i}: {'    ,   '.join(result.call_steps)}")

            if result.is_final_step:
                final_result = result.results[-1][0]   # Last result --> first element in the tuple
                # print(f"Final result: {final_result}")
                break

            steps.extend(result.call_steps)

            prompt_msg = self._prepare_next_prompt(prompt_msg, expression, steps, result.results, response)

            if i >= self.max_llm_calls:
                raise RuntimeError(f'Max LLM calls reached before final result. Max calls: {self.max_llm_calls}')

            i += 1

        return final_result

    def _process_tool_calls(self, tool_calls: List[Any]) -> ToolCallResult:
        if not tool_calls:
            raise RuntimeError("Error: Expected a tool call but received none.")

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

            step = f"{a} {op} {b} = {result}"
            call_steps.append(step)

            results.append((result, tool_call.id))

        return ToolCallResult(results, is_final_step, call_steps, '')

    def _prepare_next_prompt(self, prompt_msg: MessageHistory, expression: str, steps: List[str],
                             results: List[Tuple[float, str]], response: Any) -> MessageHistory:
        # next_prompt = "Proceed with the next step of the calculation, in the correct order of operations."

        steps_so_far = '\n'.join(steps)
        next_prompt = self.subsequent_prompt.replace('{EXPRESSION}', expression)
        next_prompt = next_prompt.replace('{STEPS_SO_FAR}', steps_so_far)

        # print(next_prompt)

        # Append messages to the current history
        if self.append_messages:
            if self.return_tool_call_msgs:
                prompt_msg.add_generic_message(response)

                for (result, tool_call_id) in results:
                    prompt_msg.add_tool_result_message(result, tool_call_id)

            prompt_msg.add_user_message(next_prompt)

        # Fresh message history for the next iteration
        else:
            prompt_msg = MessageHistory()
            prompt_msg.add_system_message(self.system_prompt)
            prompt_msg.add_user_message(next_prompt)

        return prompt_msg


