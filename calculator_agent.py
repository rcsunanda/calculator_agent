from pprint import pprint

import openai
import os
import json

from calculator import calculate


api_key = os.environ.get("OPEN_AI_TOKEN")


class CalculatorAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = 'gpt-3.5-turbo'

        self.system_prompt = "You are a calculator agent. Given a string describing a mathematical expression, " \
                             "you can determine the next *single* calculation step to be performed in the form of a function call to a calculate function. " \
                             "Each calculate step is specified by two numbers (a, b) and an operation (op). " \
                             "The four valid operations are '+' for addition, '-' for subtraction, '*' for multiplication, '/' for division" \
                             "The answer to each step will be calculated using a calculate function and given back to you." \
                             "At each step, output the next *single* function call necessary to calculate the result."

    def run(self, expression: str):
        print(f"Input expression: {expression}")

        initial_prompt = f'This is the mathematical expression to be evaluated: {expression}. ' \
                 f'Perform the calculation step by step, making tool calls to the provided function.'

        tools = [
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

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_prompt},
        ]

        steps = []
        final_result = None

        while True:
            print('\n----- messages -----\n')
            print(messages)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )

            response = completion.choices[0].message

            if not response.tool_calls:
                print("Error: Expected a function call but received none.")
                break

            # Keep only the first element in the list, as we are expecting only one function call at a time
            response.tool_calls = response.tool_calls[:1]

            function_call = response.tool_calls[0].function
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
            steps.append(step)
            print(f"Step: {step}")

            if is_final_step:
                final_result = result
                print(f"Final result: {final_result}")
                break

            function_call_result_message = {
                "role": "tool",
                "content": json.dumps({"result": result}),
                "tool_call_id": response.tool_calls[0].id
            }

            next_prompt = "Proceed with the next step of the calculation, in the correct order of operations."

            # steps_so_far = '\n'.join(steps)
            # next_prompt = f"Proceed with the next step of the calculation. For reference, the original expression is: {expression}. " \
            #               f"And the steps calculated so far are: {steps_so_far}."

            messages.extend([
                response,
                function_call_result_message,
                {"role": "user", "content": next_prompt},
            ])

        # print("Calculation steps:")
        # for step in steps:
        #     print(step)
        # print(f"Final result: {final_result}")

        return final_result

    def _validate_input(self, input_str):
        pass


