import json

from src.tools.calculator import calculate
from src.llm.chatgpt import MessageHistory


class StepwiseCalculatorAgent:
    def __init__(self, llm_client, config):
        self.llm_client = llm_client

        self.system_prompt = config['system_prompt']
        self.subsequent_prompt = config['subsequent_prompt']
        self.initial_prompt = config['initial_prompt']

        self.max_calls = config['max_llm_calls']
        self.return_tool_call_msgs = config['return_tool_call_msgs']
        self.append_messages = config['append_messages']

    def run(self, expression: str):
        print(f"Input expression: {expression}")

        initial_prompt = self.initial_prompt.replace('{EXPRESSION}', expression)

        prompt_msg = MessageHistory()
        prompt_msg.add_system_message(self.system_prompt)
        prompt_msg.add_user_message(initial_prompt)

        steps = []
        i = 1
        final_result = None

        while True:
            print(f'\n ================ iteration {i} ================ \n')

            print('\n----- prompt_msg -----\n')
            print(prompt_msg)

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


