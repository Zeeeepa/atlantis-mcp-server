import atlantis

from openai import OpenAI




@chat
async def chat(address:str):
    """
    This is a placeholder function for 'chat'
    """
    print(f"Executing placeholder function: chat...")

    await atlantis.client_log("chat running")

    # Configure client to use local server
    client = OpenAI(
        # base_url="http://localhost:11434/v1",  # Point to ollama server
        base_url="http://100.106.176.121:10240/v1",  # Point to mlx omni server
        api_key="not-needed",  # API key is not required for local server
    )

    model = "mlx-community/Qwen3-30B-A3B-8bit"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{
        "role": "user",
        "content": "What's the weather like in Boston today?"
    }]

    completion = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto"
    )

    message = completion.choices[0].message
    print(f"message: {message}")
    if message.tool_calls:
        print("success")
    else:
        print("failed")

    return message


