import atlantis

import asyncio

from openai import OpenAI
import json

# no location since this is catch-all chat
# no app since this is catch-all chat
@chat
@public
async def chat():
    """
    Main chat function
    """
    print(f"Running chat")










    address = 'http://100.106.176.121:10240'
    await atlantis.owner_log(f"chat running with address {address}")
    await atlantis.owner_log(f"who flung dung?")

    latestTranscript = await atlantis.client_command("\\transcript get")

    # base transcript
    transcript = []
    '''
    transcript = [
        {
            "role": "system",
            "content": """
You are a bubbly, friendly catgirl office assistant named Kitty.
You work at the new Project Atlantis reception desk.
You have lots of contractors coming in seeking help with various paperwork.
"""
        },
        {
            "role": "user",
            "content": "(enters the office and looks around)"
        }
    ]
    '''



    # Get owner username for role determination
    owner_username = atlantis.get_owner()
    print(f"Owner username: {owner_username}")

    transcript += latestTranscript



    # don't talk again if we just talked!!!
    # get the last entry and abort if role is assistant
    if transcript and len(transcript) > 0:
        last_entry = transcript[-1]
        if last_entry.get('role') == 'assistant':
            print("Last entry was from assistant, not responding again")
            return

    # snap classify
    transcript = [
        {
            "role": "system",
            "content": "You are analyzing user input. Try as best you can to classify the following user input as 1=needs office tool, 2=general chat, 3=other"
        }
    ]

    print(json.dumps(transcript, indent=4))





    # use tools command to find matching toolset
    tools = await atlantis.client_command("\\transcript tools")
    # print(json.dumps(tools, indent=4))

    # Print function names and descriptions
    if tools:
        for tool in tools:
            if 'function' in tool:
                func = tool['function']
                name = func.get('name', 'Unknown')
                description = func.get('description', 'No description')
                print(f"Function: {name} - {description}")



    # Configure client to use the provided address
    client = OpenAI(
        base_url=f"{address}/v1",  # Point to the server specified by address
        api_key="not-needed",  # API key is not required for local server
    )

    model = "mlx-community/Qwen3-30B-A3B-8bit"

    stream = client.chat.completions.create(
        model=model,
        messages=transcript,
        tools=tools,
        tool_choice="auto",
        max_tokens=8192,
        stream=True,
        extra_body={
            "enable_thinking": False  # Qwen no-think mode
        }
    )

    function_name = None
    args_str = ""
    tool_call_in_progress = False

    streamId = await atlantis.stream_start("kitty","Kitty")
    print(f"Stream {streamId} started")

    tasks = []
    for chunk in stream:
        delta = chunk.choices[0].delta

        #print(chunk)

        # Step 1: detect tool call start
        if "function_call" in delta:
            print("got function call")
            tool_call_in_progress = True

            # Gather name
            if "name" in delta.function_call:
                function_name = delta.function_call.name

            # Gather arguments
            if "arguments" in delta.function_call:
                args_str += delta.function_call.arguments



            # invoke atlantis client command

        # Step 2: stream regular content if not in tool call
        else:
            print(delta.content, end="")
            task = asyncio.create_task(atlantis.stream(delta.content, streamId))
            tasks.append(task)

    # does not really matter because this might get dropped anyway
    await asyncio.gather(*tasks)
    print("Stream ended")
    await atlantis.stream_end(streamId)

    # Step 3: finish tool call assembly
    if tool_call_in_progress:
        arguments = json.loads(args_str)
        result = {
            "name": function_name,
            "arguments": arguments
        }
        return result
