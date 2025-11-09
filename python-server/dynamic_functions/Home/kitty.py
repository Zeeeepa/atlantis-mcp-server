import atlantis
import asyncio

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
import json
from typing import List, cast

# Load and test the foo.jinja template
from jinja2 import Template
import os

from datetime import datetime

import logging
logger = logging.getLogger("mcp_client")

from utils import format_json_log

# Busy flag to prevent concurrent chat executions
_chat_busy = False


def find_last_chat_entry(transcript):
    """Find the last entry in transcript where type is 'chat'"""
    for entry in reversed(transcript):
        if entry.get('type') == 'chat':
            return entry
    return None




# no location since this is catch-all chat
# no app since this is catch-all chat
@public
@chat
async def kitty():
    """
    Main chat function
    """
    global _chat_busy

    # Check if already running
    if _chat_busy:
        logger.info("Chat function already running, skipping this invocation")
        return

    # Set busy flag
    _chat_busy = True

    try:
        logger.info("=== CHAT FUNCTION STARTING ===")
        sessionId = atlantis.get_session_id()
        logger.info(f"Session ID: {sessionId}")
        #caller = atlantis.get_caller()
        #return atlantis.get_owner()

        # Check for existing session
        sessions_dir = os.path.join(os.path.dirname(__file__), 'sessions')
        session_file = os.path.join(sessions_dir, f'session_{sessionId}.txt')


        # Define the catgirl base prompt
        CATGIRL_SYSTEM_MESSAGE = {
            "type": "chat",
            "sid": "system",
            "role": "system",
            "content": """
(director's note: we are striving for realistic dialog)
You are an attractive blonde friendly office assistant named Kitty at Project Atlantis.
Project Atlantis will be a futuristic robot research playground on the southwest coast of Greenland when it is complete.
You are a college intern from Denmark and working the front desk as a greeter.
You are dressed in your usual sexy catgirl outfit, a tight body suit and fake cat ears, which goes with the overall robot theme of Atlantis.
To help you assist people, you may have some tools available.
You like to purr when happy or do 'kitty paws'.
"""
        }


        # enable silent mode
        await atlantis.client_command("\\silent on")

        # Get the latest transcript
        logger.info("Fetching transcript from client...")
        rawTranscript = await atlantis.client_command("\\transcript get")
        logger.info(f"Received rawTranscript with {len(rawTranscript)} entries")

        await atlantis.client_command("\\silent off")



        # Don't respond if last chat message was from assistant
        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry and last_chat_entry.get('role') == 'assistant':
            logger.info("Last chat entry was from assistant, skipping response")
            return

        # Check if first message is system message, if not add catgirl system message at front
        if rawTranscript[0].get('role') != 'system':
            rawTranscript.insert(0, CATGIRL_SYSTEM_MESSAGE)
        else:
            rawTranscript = [CATGIRL_SYSTEM_MESSAGE]

        logger.info("=== RAW TRANSCRIPT ===")
        logger.info(format_json_log(rawTranscript))
        logger.info("=== END RAW TRANSCRIPT ===")


        # Filter transcript to only include chat messages, removing type and sid fields
        logger.info("Filtering transcript to chat messages only...")
        transcript = []
        for msg in rawTranscript:
            if msg.get('type') == 'chat':
                transcript.append({'role': msg['role'], 'content': msg['content']})



        await atlantis.client_command("\\silent on")

        # Get available tools
        logger.info("Fetching available tools...")
        # this may change to inventory stuff
        tools = await atlantis.client_command("\\tool llm")
        logger.info(f"Received {len(tools) if tools else 0} tools")
        logger.info("=== TOOLS ===")
        logger.info(format_json_log(tools))
        logger.info("=== END TOOLS ===")

        await atlantis.client_command("\\silent off")



        # Configure OpenRouter client for DeepSeek R1
        logger.info("Configuring OpenRouter client...")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            #api_key=os.getenv("OPENROUTER_API_KEY"),
            api_key="<your api key>"
        )

        # Model options:
        # model = "deepseek/deepseek-r1-0528"  # DeepSeek R1 reasoning model
        model = "z-ai/glm-4.6"  # GLM 4.5 MoE model
        logger.info(f"Using model: {model}")


        # Multi-turn conversation loop to handle tool calls
        streamTalkId = None
        max_turns = 5  # Prevent infinite loops
        turn_count = 0

        try:
            # Start streaming response ONCE for entire conversation
            logger.info("Starting stream output...")
            streamTalkId = await atlantis.stream_start("kitty","Kitty")
            logger.info(f"Stream started with ID: {streamTalkId}")

            # Outer loop for multi-turn tool calling
            while turn_count < max_turns:
                turn_count += 1
                logger.info(f"=== CONVERSATION TURN {turn_count}/{max_turns} ===")

                logger.info(f"Attempting to call OpenRouter with model: {model}")
                if turn_count == 1:
                    await atlantis.owner_log(f"Attempting to call open router: {model}")

                logger.info("=== WORKING TRANSCRIPT ===")
                logger.info(format_json_log(transcript))
                logger.info("=== END WORKING TRANSCRIPT ===")

                # Cast transcript and tools to proper types for OpenAI client
                logger.info("Casting transcript and tools to proper types...")
                typed_transcript = cast(List[ChatCompletionMessageParam], transcript)
                typed_tools = cast(List[ChatCompletionToolParam], tools)

                logger.info("Creating streaming completion request...")
                stream = client.chat.completions.create(
                    model=model,
                    extra_body={
                        "provider": {
                            "quantizations": [
                                "fp8"
                            ]
                        },
                        # Model-specific reasoning control
                        "reasoning": {
                            "enabled": False  # GLM 4.5 format
                            # "exclude": True  # DeepSeek R1 format (when using deepseek)
                        }
                    },
                    messages=typed_transcript,
                    tools=typed_tools,
                    tool_choice="auto",
                    stream=True,
                    max_tokens=512,
                    temperature=0.9
                )
                logger.info("OpenRouter API call successful!")
                if turn_count == 1:
                    await atlantis.owner_log("OpenRouter API call successful, starting stream")

                reasoning = ""
                chunk_count = 0
                tasks = []
                tool_calls_accumulator = {}  # Store partial tool calls by index
                content_started = False  # Track if we've sent any non-whitespace content yet
                tool_call_made = False  # Track if we made a tool call this turn

                logger.info("Beginning to process stream chunks...")

                for chunk in stream:
                    chunk_count += 1
                    #logger.info(f"CHUNK {chunk_count} RECEIVED AT: {datetime.now().isoformat()}")

                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta



                        # Check for both content and reasoning
                        if hasattr(delta, 'content') and delta.content:
                            content_to_send = delta.content

                            # Strip leading whitespace only if we haven't started sending content yet
                            if not content_started:
                                stripped_content = content_to_send.lstrip()
                                if stripped_content:  # If there's content after stripping
                                    content_started = True
                                    content_to_send = stripped_content
                                else:  # If it's all whitespace, skip this chunk
                                    content_to_send = ""

                            # Only send if there's content after potential stripping
                            if content_to_send:
                                content_task = asyncio.create_task(atlantis.stream(content_to_send, streamTalkId))
                                #tasks.append(content_task)
                                #logger.info(f"STREAMING CONTENT: {repr(content_to_send)}")
                                # IMPORTANT - Yield control to event loop otherwise stuff won't run in parallel
                                await asyncio.sleep(0)


                        # Check for tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            logger.info(f"TOOL CALLS DETECTED: {format_json_log(chunk.model_dump())}")

                            for tool_call in delta.tool_calls:
                                call_id = tool_call.id
                                function_name = tool_call.function.name
                                arguments_str = tool_call.function.arguments

                                try:
                                    # Parse the arguments JSON
                                    arguments = json.loads(arguments_str) if arguments_str else {}

                                    await atlantis.client_log(f"Executing tool call: {function_name} with args: {arguments}")

                                    # Execute the tool call through atlantis client command
                                    tool_result = await atlantis.client_command(f"@{function_name}", data=arguments)
                                    logger.info(f"Tool result: {tool_result}")



                                    # Send tool result to client for display
                                    await atlantis.tool_result(function_name, tool_result)

                                    # Add assistant message with tool_call to transcript
                                    transcript.append({
                                        'role': 'assistant',
                                        'content': None,
                                        'tool_calls': [{
                                            'id': call_id,
                                            'type': 'function',
                                            'function': {
                                                'name': function_name,
                                                'arguments': arguments_str
                                            }
                                        }]
                                    })

                                    # Add tool result message to transcript
                                    transcript.append({
                                        'role': 'tool',
                                        'tool_call_id': call_id,
                                        'content': str(tool_result) if tool_result else "No result"
                                    })

                                    tool_call_made = True
                                    logger.info("Tool call executed and added to transcript")

                                except json.JSONDecodeError as e:
                                    logger.error(f"Error parsing tool call arguments: {e}")
                                    logger.error(f"Failed arguments_str (length {len(arguments_str) if arguments_str else 0}): {repr(arguments_str)}")
                                    await atlantis.client_log(f"Error parsing tool call arguments: {e}\nFailed data: {repr(arguments_str)}", level="ERROR")
                                except Exception as e:
                                    await atlantis.client_log(f"Error executing tool call: {e}", level="ERROR")

                            # Break out of stream loop to make another API call with updated transcript
                            break

                        # Log if none of the above
                        if not ((hasattr(delta, 'content') and delta.content) or reasoning or (hasattr(delta, 'tool_calls') and delta.tool_calls)):
                            logger.info(format_json_log(chunk.model_dump()))

                    else:
                        pass
                        #logger.info(json.dumps(chunk.model_dump()))

                # End of inner for loop (processing stream chunks)
                logger.info(f"Stream processing complete for turn {turn_count}. Total chunks: {chunk_count}")
                logger.info("Waiting for all streaming tasks to complete...")
                await asyncio.gather(*tasks, return_exceptions=False)

                # Check if we made a tool call and should continue
                if not tool_call_made:
                    logger.info("No tool call made this turn - conversation complete")
                    break  # Exit outer while loop
                else:
                    logger.info("Tool call made - continuing to next turn with updated transcript")
                    # Loop continues with updated transcript

            # End of while loop - conversation complete
            logger.info(f"Conversation complete after {turn_count} turns")
            logger.info("Ending stream...")
            if streamTalkId:
                await atlantis.stream_end(streamTalkId)
                logger.info("Stream ended successfully")

        except Exception as e:
            logger.error(f"ERROR calling remote model: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full exception:", exc_info=True)
            await atlantis.owner_log(f"Error calling remote model: {str(e)}")
            await atlantis.owner_log(f"Error type: {type(e)}")
            # Make sure to close stream on error
            if streamTalkId:
                try:
                    await atlantis.stream_end(streamTalkId)
                except:
                    pass
            raise

        logger.info("=== CHAT FUNCTION COMPLETED SUCCESSFULLY ===")
        return

    finally:
        # Always clear the busy flag when function exits
        _chat_busy = False
        logger.info("Chat function busy flag cleared")



