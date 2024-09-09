import os
import json

from src.tools.calculator import calculate
from src.llm.chatgpt import ChatGPTClient, MessageHistory


class StepwiseCalculatorAgent:
    def __init__(self):
        api_key = os.environ.get("OPEN_AI_TOKEN_2")

        llm_config = {
            'api_key': api_key,
            'model': 'gpt-4o',      # 'gpt-3.5-turbo'
            'tool_call_required': True,
            'tools': [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform a calculation step (the operation op on the two operands a, b. "
                                       "is_final_step must be set to True for the last operation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number", "description": "First number."},
                                "b": {"type": "number", "description": "Second number."},
                                "op": {"type": "string", "enum": ["+", "-", "*", "/"],
                                       "description": "Operation to perform."},
                                "is_final_step": {"type": "boolean",
                                                  "description": "True only if this is the final step of the calculation."},
                            },
                            "required": ["a", "b", "op", "is_final_step"],
                        },
                    }
                },
            ]
        }

        self.llm_client = ChatGPTClient(llm_config)

        self.system_prompt = "You are a src agent. Given a string describing a mathematical expression, " \
                             "you can determine the next *single* calculation step to be performed in the form of a function call to a calculate function. " \
                             "Each calculate step is specified by two numbers (a, b) and an operation (op). " \
                             "The four valid operations are '+' for addition, '-' for subtraction, '*' for multiplication, '/' for division" \
                             "The answer to each step will be calculated using a calculate function and given back to you." \
                             "At each step, output the next *single* function call necessary to calculate the result."

        self.max_calls = 10
        self.return_tool_call_msgs = True
        self.append_messages = False

    def run(self, expression: str):
        print(f"Input expression: {expression}")

        initial_prompt = f'This is the mathematical expression to be evaluated: {expression}. ' \
                 f'Perform the calculation step by step, making tool calls to the provided function.'

        prompt_msg = MessageHistory()
        prompt_msg.add_system_message(self.system_prompt)
        prompt_msg.add_user_message(initial_prompt)

        steps = []
        i = 1
        final_result = None

        while True:
            print(f'\n ================ iteration {i} ================ \n')
            # print('\n----- messages -----\n')
            # print(messages)

            response = self.llm_client.run_prompt(prompt_msg)

            results, is_final_step, call_steps = self._process_tool_calls(response.tool_calls)

            print(f"Call {i}: {'    ,   '.join(call_steps)}")

            if is_final_step:
                final_result = results[-1][0]   # Last result --> first element in the tuple
                print(f"Final result: {final_result}")
                break

            steps.extend(call_steps)

            prompt_msg = self._prepare_next_prompt(prompt_msg, expression, steps, results, response)

            if i >= self.max_calls:
                print(f"Max calls reached: {self.max_calls}")
                break

            i += 1

        return final_result

    def _validate_input(self, input_str):
        pass

    def _process_tool_calls(self, tool_calls):
        if not tool_calls:
            raise ValueError("Error: Expected a function call but received none.")

        results = []
        is_final_step = False
        call_steps = []

        for tool_call in tool_calls:
            function_call = tool_call.function
            func_args = json.loads(function_call.arguments)

            a = func_args['a']
            b = func_args['b']
            op = func_args['op']
            is_final_step = func_args['is_final_step']

            result = calculate(a, b, op)

            step = f"{a} {op} {b} = {result}"
            call_steps.append(step)

            results.append((result, tool_call.id))

        return results, is_final_step, call_steps

    def _prepare_next_prompt(self, prompt_msg, expression, steps, results, response):
        # next_prompt = "Proceed with the next step of the calculation, in the correct order of operations."

        steps_so_far = '\n'.join(steps)
        next_prompt = f"Proceed with the next step of the calculation. For reference, the original expression is: \n {expression}. " \
                      f"And the steps calculated so far are: \n {steps_so_far}."

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


