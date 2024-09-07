from pprint import pprint

import openai
import os
import json

from calculator import calculate


api_key = os.environ.get("OPEN_AI_TOKEN")


class CalculatorAgent():
    def __init__(self):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = 'gpt-3.5-turbo'

        self.system_prompt = "You are a calculator agent. Given a string describing a mathematical expression, " \
                             "you can determine the next *single* calculation step to be performed in the form of a function call to a calculate function. " \
                             "Each calculate step is specified by two numbers (a, b) and an operation (op). " \
                             "The four valid operations are '+', '-', '*', '/'" \
                             "The answer to each step will be calculated using a calculate function and given back to you." \
                             "At each step, output the next *single* function call necessary to calculate the result."

    def run(self, input_str):
        prompt = f'Following is a mathematical expression to be evaluated. ' \
                 f'What is the next function call to be performed? {input_str}'

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Given two numbers, and an operation (op), perform the mathematical operation on the two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "First number.",
                            },

                            "b": {
                                "type": "number",
                                "description": "Second number.",
                            },

                            "op": {
                                "type": "string",
                                "description": "Operation to perform on the two numbers.",
                            },
                        },
                        "required": ["a", "b", "op"],
                        "additionalProperties": False,
                    },
                }
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            tools=tools,
        )

        print('\n----- Completion -----\n')
        pprint(completion)

        print('\n----- Completion message -----\n')
        pprint(completion.choices[0].message)

        response = completion.choices[0].message.tool_calls[0].function.arguments

        print('\n----- tool_calls -----\n')
        pprint(completion.choices[0].message.tool_calls)

        func_args = json.loads(response)

        print('\n----- func_args -----\n')
        pprint(func_args)

        a = func_args['a']
        b = func_args['b']
        op = func_args['op']

        ans = calculate(a, b, op)

        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({"a": a, "b": b, "op": op, 'result': ans}),
            "tool_call_id": completion.choices[0].message.tool_calls[0].id
        }

        print('\n----- function_call_result_message -----\n')
        print(function_call_result_message)

        print('\n ********* Sending next prompt *********\n')

        prompt_2 = f'What is the next function call to be performed?'

        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                completion.choices[0].message,
                function_call_result_message,
                {"role": "user", "content": prompt_2},
            ]

        print('\n----- messages -----\n')
        print(messages)

        completion_2 = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )

        print('\n----- Completion message -----\n')
        pprint(completion_2.choices[0].message)

        response_2 = completion_2.choices[0].message.tool_calls[0].function.arguments

        print('\n----- tool_calls_2 -----\n')
        pprint(completion_2.choices[0].message.tool_calls)

        func_args_2 = json.loads(response_2)

        print('\n----- func_args_2 -----\n')
        pprint(func_args_2)

        a = func_args_2['a']
        b = func_args_2['b']
        op = func_args_2['op']

        ans = calculate(a, b, op)

        return ans

    def _validate_input(self, input_str):
        pass


