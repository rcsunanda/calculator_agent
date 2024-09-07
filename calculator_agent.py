from pprint import pprint

import openai
import os
import json

from calculator import calculate


class CalculatorAgent:
    def __init__(self):
        api_key = os.environ.get("OPEN_AI_TOKEN_2")

        self.client = openai.OpenAI(api_key=api_key)
        # self.model = 'gpt-3.5-turbo'
        self.model = 'gpt-4o'

        self.system_prompt = "You are a calculator agent. Given a string describing a mathematical expression, " \
                             "you can determine the next *single* calculation step to be performed in the form of a function call to a calculate function. " \
                             "Each calculate step is specified by two numbers (a, b) and an operation (op). " \
                             "The four valid operations are '+' for addition, '-' for subtraction, '*' for multiplication, '/' for division" \
                             "The answer to each step will be calculated using a calculate function and given back to you." \
                             "At each step, output the next *single* function call necessary to calculate the result."

        self.max_calls = 3
        self.return_tool_call_msgs = True

        self.tools = [
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

    def run(self, expression: str):
        print(f"Input expression: {expression}")

        initial_prompt = f'This is the mathematical expression to be evaluated: {expression}. ' \
                 f'Perform the calculation step by step, making tool calls to the provided function.'

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_prompt},
        ]

        steps = []
        i = 0
        final_result = None

        while True:
            i += 1
            # print('\n----- messages -----\n')
            # print(messages)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="required",
            )

            response = completion.choices[0].message

            if not response.tool_calls:
                print("Error: Expected a function call but received none.")
                break

            function_call_result_message = []

            is_final_step = False
            result = None
            call_steps = []

            for tool_call in response.tool_calls:
                function_call = tool_call.function
                func_args = json.loads(function_call.arguments)

                a = func_args['a']
                b = func_args['b']
                op = func_args['op']
                is_final_step = func_args['is_final_step']

                try:
                    result = calculate(a, b, op)
                except Exception as e:
                    print(f"Error in calculation: {e}")
                    return str(e)

                step = f"{a} {op} {b} = {result}"
                call_steps.append(step)
                steps.append(step)

                function_call_result_message.append({
                    "role": "tool",
                    "content": json.dumps({"result": result}),
                    "tool_call_id": tool_call.id
                })

            print(f"Call {i}: {'    ,   '.join(call_steps)}")

            if is_final_step:
                final_result = result if result is not None else final_result
                print(f"Final result: {final_result}")
                break

            # next_prompt = "Proceed with the next step of the calculation, in the correct order of operations."

            steps_so_far = '\n'.join(steps)
            next_prompt = f"Proceed with the next step of the calculation. For reference, the original expression is: {expression}. " \
                          f"And the steps calculated so far are: {steps_so_far}."

            # Keep only the first element in the list, as we expect & handle only one function call at a time
            # response.tool_calls = response.tool_calls[:1]

            if self.return_tool_call_msgs:
                messages.extend([
                    response,
                    *function_call_result_message,
                ])

            messages.append({"role": "user", "content": next_prompt})

            if i >= self.max_calls:
                print(f"Max calls reached: {self.max_calls}")
                break

        # print("Calculation steps:")
        # for step in steps:
        #     print(step)
        # print(f"Final result: {final_result}")

        return final_result

    def _validate_input(self, input_str):
        pass


