import openai
import json


class MessageHistory:
    def __init__(self, messages=None):
        if messages is None:
            messages = []

        self.messages = messages

    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result_message(self, result, tool_call_id):
        self.messages.append({
            "role": "tool",
            "content": json.dumps({"result": result}),
            "tool_call_id": tool_call_id
        })

    def add_generic_message(self, msg):
        self.messages.append(msg)

    def get_messages(self):
        return self.messages

    # def __str__(self):
    #     return str(self.messages)

    def __repr__(self):
        return str(self.messages)


class ChatGPTClient:
    def __init__(self, config):
        self.client = openai.OpenAI(api_key=config['api_key'])

        self.model = config['model']
        self.tool_definitions = config['tool_definitions']
        self.tool_call_required = config['tool_call_required']

    def run_prompt(self, msg_history):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=msg_history.get_messages(),
            tools=self.tool_definitions,
            tool_choice=self.tool_call_required,
        )

        response = completion.choices[0].message
        return response
