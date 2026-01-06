import atlantis
import asyncio

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
import json
from typing import List, Dict, Any, Optional, TypedDict, Tuple, cast


# =============================================================================
# Cloud Tool Types (input from cloud)
# =============================================================================

class ToolT(TypedDict, total=False):
    """Tool record from the cloud"""
    remote_id: int
    tool_id: int
    perm_id: int
    app_name: str
    remote_user_id: int

    is_chat: bool
    is_tick: bool
    is_session: bool
    is_game: bool
    is_index: bool
    is_public: bool

    is_connected: bool
    is_default: bool

    hostname: str
    port: int

    remote_owner: str
    remote_name: str

    mcp_name: str
    mcp_tool: str

    tool_app: str
    tool_location: str
    tool_name: str
    protection_name: str
    tool_type: str
    tool_description: str
    filename: str

    price_per_call: float
    price_per_sec: float

    static_error_msg: str
    runtime_error_msg: str
    params: str
    input_schema: str

    started_at: str  # ISO date string
    remote_updated_at: str  # ISO date string


# =============================================================================
# LLM Tool Types (output for OpenAI-compatible APIs)
# =============================================================================

class ToolSchemaPropertyT(TypedDict, total=False):
    """Property definition within a tool schema"""
    type: str
    description: str
    enum: List[str]


class ToolSchemaT(TypedDict, total=False):
    """JSON Schema for tool parameters"""
    type: str
    properties: Dict[str, ToolSchemaPropertyT]
    required: List[str]


class ToolFunctionT(TypedDict):
    """Function definition within a tool"""
    name: str
    description: str
    parameters: ToolSchemaT


class TranscriptToolT(TypedDict):
    """Tool format for LLM transcripts (OpenAI-compatible)"""
    type: str
    function: ToolFunctionT


class SimpleToolT(TypedDict, total=False):
    """Simplified tool format"""
    type: str
    name: str
    description: str
    parameters: ToolSchemaT


# =============================================================================
# Tool Conversion Functions
# =============================================================================

def get_consolidated_full_name(tool: ToolT) -> str:
    """
    Build a consolidated tool name from tool metadata.
    Format: remote_owner*remote_name*app*location*function
    Only includes parts that are needed to disambiguate.
    """
    # Extract values with .get() to satisfy Pylance
    remote_owner = tool.get('remote_owner', '')
    remote_name = tool.get('remote_name', '')
    tool_app = tool.get('tool_app', '')
    tool_location = tool.get('tool_location', '')
    tool_name = tool.get('tool_name', '')

    parts = [remote_owner, remote_name, tool_app, tool_location, tool_name]

    # Build the full name, but simplify if possible
    # If all prefix parts are empty, just return the tool name
    if all(p == '' for p in parts[:-1]):
        return parts[-1]

    return '*'.join(parts)


def convert_tools_for_llm(
    tools: List[ToolT],
    show_hidden: bool = False
) -> Tuple[List[TranscriptToolT], List[SimpleToolT]]:
    """
    Convert cloud tool records to LLM-compatible format.

    Args:
        tools: List of ToolT records from the cloud
        show_hidden: If False, skip tools starting with '_'

    Returns:
        Tuple of (full tools list, simple tools list)
    """
    out_tools: List[TranscriptToolT] = []
    out_tools_simple: List[SimpleToolT] = []

    for tool in tools:
        # Skip hidden tools unless show_hidden is True
        tool_name = tool.get('tool_name', '')
        if not show_hidden and tool_name.startswith('_'):
            continue

        # Skip chat, tick, and session tools
        if tool.get('is_chat') or tool.get('is_tick') or tool.get('is_session'):
            continue

        # Only process function, internal, or mcp_tool types
        tool_type = tool.get('tool_type', '')
        if tool_type not in ('function', 'internal', 'mcp_tool'):
            continue

        # Parse the input schema
        input_schema_str = tool.get('input_schema', '{}')
        try:
            schema = cast(ToolSchemaT, json.loads(input_schema_str))
        except json.JSONDecodeError:
            schema: ToolSchemaT = {'type': 'object', 'properties': {}, 'required': []}

        # Get consolidated name
        consolidated_name = get_consolidated_full_name(tool)

        # Build the output tool
        out_tool: TranscriptToolT = {
            'type': 'function',
            'function': {
                'name': consolidated_name,
                'description': tool.get('tool_description', ''),
                'parameters': schema
            }
        }

        out_tools.append(out_tool)

        # Build the simple version
        out_tools_simple.append({
            'type': 'function',
            'name': consolidated_name,
            'description': tool.get('tool_description', ''),
            'parameters': schema
        })

    return out_tools, out_tools_simple

# Load and test the foo.jinja template
from jinja2 import Template
import os

from datetime import datetime

import logging
logger = logging.getLogger("mcp_client")

from utils import format_json_log


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

    Note: Concurrent calls are automatically queued by the server,
    so each invocation runs sequentially without manual busy flag management.
    """
    logger.info("=== CHAT FUNCTION STARTING ===")
    sessionId = atlantis.get_session_id()
    logger.info(f"Session ID: {sessionId}")
    #caller = atlantis.get_caller()
    #return atlantis.get_owner()

    # Check for existing session
    sessions_dir = os.path.join(os.path.dirname(__file__), 'sessions')
    session_file = os.path.join(sessions_dir, f'session_{sessionId}.txt')

    # The rest of the function body is indented due to the removed try/finally block
    # Keeping the indentation to avoid reformatting the entire file
    if True:
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
You are dressed in your usual sexy catgirl outfit, a tight white body suit and fake cat ears, which goes with the overall robot theme of Atlantis.
To help you assist people, you may have some tools available.
You can also explain that things are a bit of a mess right now but new terminal users can use '/help' command.
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



        # Don't respond if last chat message was from Kitty (the bot)
        # Note: We check 'sid' not 'role' because user messages also have role='assistant'
        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == 'kitty':
            logger.warning("\x1b[38;5;204mLast chat entry was from kitty (bot), skipping response\x1b[0m")
            return

        # Validate transcript before processing
        if not rawTranscript:
            logger.error("!!! CRITICAL: rawTranscript is EMPTY - no messages received from client!")
            await atlantis.owner_log("CRITICAL: rawTranscript is empty!")
            raise ValueError("Cannot process empty transcript")

        logger.info(f"rawTranscript has {len(rawTranscript)} entries before system message handling")

        # Check if first message is system message, if not add catgirl system message at front
        if rawTranscript[0].get('role') != 'system':
            rawTranscript.insert(0, CATGIRL_SYSTEM_MESSAGE)
            logger.info("Inserted catgirl system message at front of transcript")
        else:
            # Replace existing system message with catgirl's system message
            logger.warning(f"!!! Replacing existing system message. Old: {rawTranscript[0].get('content', '')[:100]}...")
            rawTranscript[0] = CATGIRL_SYSTEM_MESSAGE

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
        tools = await atlantis.client_command("\\dir")
        logger.info(f"Received {len(tools) if tools else 0} tools")
        logger.info("=== TOOLS ===")
        logger.info(format_json_log(tools))
        logger.info("=== END TOOLS ===")

        await atlantis.client_command("\\silent off")


        # uses env var
        # Configure OpenRouter client for DeepSeek R1
        logger.info("Configuring OpenRouter client...")

        # Check for API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            error_msg = "OPENROUTER_API_KEY environment variable is not set"
            logger.error(error_msg)
            await atlantis.owner_log(error_msg)
            raise ValueError(error_msg)

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
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

                # Cast transcript and tools to proper types for OpenAI client
                logger.info("Casting transcript and tools to proper types...")
                typed_transcript = cast(List[ChatCompletionMessageParam], transcript)

                # Convert tools from cloud format to OpenAI-compatible format
                converted_tools, _ = convert_tools_for_llm(tools)
                typed_tools = cast(List[ChatCompletionToolParam], converted_tools)

                # Log what we're actually sending to OpenRouter
                logger.info(f"=== SENDING TO OPENROUTER (turn {turn_count}) ===")
                logger.info(f"Messages ({len(typed_transcript)} entries):")
                logger.info(format_json_log(typed_transcript))
                logger.info(f"Tools ({len(typed_tools)} entries):")
                logger.info(format_json_log(typed_tools))
                logger.info("=== END OPENROUTER PAYLOAD ===")

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
                                index = tool_call.index

                                # Initialize accumulator for this tool call if needed
                                if index not in tool_calls_accumulator:
                                    tool_calls_accumulator[index] = {
                                        'id': None,
                                        'name': None,
                                        'arguments': ''
                                    }

                                # Accumulate the chunks
                                if tool_call.id:
                                    tool_calls_accumulator[index]['id'] = tool_call.id
                                if tool_call.function.name:
                                    tool_calls_accumulator[index]['name'] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls_accumulator[index]['arguments'] += tool_call.function.arguments

                                logger.info(f"Accumulated tool call {index}: id={tool_calls_accumulator[index]['id']}, name={tool_calls_accumulator[index]['name']}, args_len={len(tool_calls_accumulator[index]['arguments'])}")

                        # Check if stream is ending (finish_reason is set)
                        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                        if finish_reason == 'tool_calls':
                            logger.info("Stream finished with tool_calls - executing accumulated tool calls")

                            # Execute all accumulated tool calls
                            for index, accumulated_call in tool_calls_accumulator.items():
                                call_id = accumulated_call['id']
                                function_name = accumulated_call['name']
                                arguments_str = accumulated_call['arguments']

                                try:
                                    # Parse the complete arguments JSON
                                    arguments = json.loads(arguments_str) if arguments_str else {}

                                    #await atlantis.client_log(f"Executing tool call: {function_name} with args: {arguments}")

                                    # Execute the tool call through atlantis client command

                                    await atlantis.client_command("\\silent on")
                                    # use '%' prefix instead of '@' in call to ignore any working path / force absolute path
                                    tool_result = await atlantis.client_command(f"%{function_name}", data=arguments)
                                    await atlantis.client_command("\\silent off")

                                    logger.info(f"Tool result: {tool_result}")

                                    # Send tool result to client for transcript
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
                                    logger.error(f"Error executing tool call: {e}")
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

            # Extract detailed error info from OpenAI/OpenRouter APIError
            error_details = str(e)
            if hasattr(e, 'body') and e.body:
                logger.error(f"Error body: {e.body}")
                error_details = f"{error_details} | Body: {e.body}"
            if hasattr(e, 'code') and e.code:
                logger.error(f"Error code: {e.code}")
            if hasattr(e, 'type') and e.type:
                logger.error(f"Error type attr: {e.type}")
            if hasattr(e, 'param') and e.param:
                logger.error(f"Error param: {e.param}")
            if hasattr(e, 'response') and e.response:
                try:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response text: {e.response.text}")
                    error_details = f"{error_details} | Status: {e.response.status_code} | Response: {e.response.text}"
                except:
                    pass

            await atlantis.owner_log(f"Error calling remote model: {error_details}")
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



