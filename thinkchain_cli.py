#!/usr/bin/env python3
# /// script
# dependencies = [
#     "anthropic>=0.25.0",
#     "sseclient-py",
#     "pydantic",
#     "python-dotenv",
#     "requests",
#     "beautifulsoup4",
#     "mcp",
#     "httpx",
# ]
# ///

"""
ThinkChain CLI Interface
-----------------------------------------
Command-line interface for ThinkChain featuring:
 ▸ interleaved + extended thinking
 ▸ fine‑grained tool streaming
 ▸ early interception of tool_use blocks
 ▸ multiple tool calls per assistant turn
 ▸ minimal dependencies (no rich/prompt_toolkit)
"""

import os, json, itertools
from collections import defaultdict

import anthropic              # ≥0.25.0
from sseclient import SSEClient  # pip install sseclient‑py
from dotenv import load_dotenv

# Import our tool discovery system
from tool_discovery import (
    get_tool_discovery, get_claude_tools, execute_tool_sync, 
    refresh_tools, list_tools, initialize_tool_discovery
)

# Load environment variables from .env file
load_dotenv()

# ────────────────────────────── 0. Config ────────────────────────────── #
# Default configuration - can be changed via /config command
CONFIG = {
    "model": "claude-sonnet-4-20250514",  # or "claude-opus-4-20250514"
    "thinking_budget": 1024,  # 1024-16000
    "max_tokens": 1024
}

AVAILABLE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514"
]

BETA_HEADERS   = ",".join([
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
])

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ────────────────────────────── 1. Tools ─────────────────────────────── #
# All tools are now discovered from the tools directory and MCP servers

# Initialize tool discovery and get all available tools
print("🔍 Discovering tools...")
tool_discovery = get_tool_discovery()

# Initialize MCP integration
async def initialize_all_tools():
    total_tools = await initialize_tool_discovery()
    return total_tools

# Run initialization
try:
    import asyncio
    if hasattr(asyncio, 'get_running_loop'):
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule the initialization
            print("⚡ Scheduling MCP initialization...")
            total_tools = 0  # Will be updated later
        except RuntimeError:
            # No running loop, we can use asyncio.run
            total_tools = asyncio.run(initialize_all_tools())
            print(f"✅ Initialized {total_tools} total tools (local + MCP)")
    else:
        # Fallback for older Python versions
        total_tools = asyncio.run(initialize_all_tools())
        print(f"✅ Initialized {total_tools} total tools (local + MCP)")
except Exception as e:
    print(f"⚠️  MCP initialization failed: {e}")
    print("🔧 Continuing with local tools only")
    total_tools = len(get_claude_tools())

# Get all available tools (local + MCP)
TOOLS = get_claude_tools()
print(f"🛠️  Total tools available: {len(TOOLS)}")

# ────────────────────────────── 2. Helpers ───────────────────────────── #
def beta_headers() -> dict[str, str]:
    """Return extra headers for every call."""
    return {"anthropic-beta": BETA_HEADERS}

def run_tool(name: str, args: dict) -> str:
    """Dispatch to discovered tool function and stringify the result."""
    try:
        result = execute_tool_sync(name, args)
        return result
    except Exception as exc:
        return f"<error>Tool {name} execution failed: {exc}</error>"

# ────────────────────────────── 3. Streaming loop ────────────────────── #
def stream_once(transcript: list[dict]) -> dict:
    """
    Send one request, handle tools if any, and return the final message.
    Shows thinking process including after tool execution.
    Returns the final assistant Message object.
    """
    try:
        # Create a streaming request with thinking enabled
        with client.messages.stream(
            model=CONFIG["model"],
            max_tokens=CONFIG["max_tokens"],
            tools=TOOLS,
            messages=transcript,
            extra_headers=beta_headers(),
            thinking={"type": "enabled", "budget_tokens": CONFIG["thinking_budget"]},
        ) as stream:
            
            thinking_content = []
            in_thinking = False
            
            # Handle the stream events
            for chunk in stream:
                if chunk.type == 'message_start':
                    continue
                elif chunk.type == 'content_block_start':
                    if hasattr(chunk.content_block, 'type'):
                        if chunk.content_block.type == 'tool_use':
                            print(f"\n[tool_use:{chunk.content_block.name}] ▶", flush=True)
                        elif chunk.content_block.type == 'thinking':
                            print(f"\n[thinking] ", end='', flush=True)
                            in_thinking = True
                elif chunk.type == 'content_block_delta':
                    if hasattr(chunk.delta, 'text') and chunk.delta.text:
                        if in_thinking:
                            print(chunk.delta.text, end='', flush=True)
                            thinking_content.append(chunk.delta.text)
                        else:
                            print(chunk.delta.text, end='', flush=True)
                elif chunk.type == 'content_block_stop':
                    if in_thinking:
                        print("\n", flush=True)
                        in_thinking = False
                elif chunk.type == 'message_delta':
                    continue
                elif chunk.type == 'message_stop':
                    break
            
            # Get the final message
            final_message = stream.get_final_message()
            
            # Check if there are tool uses that need to be handled
            tool_uses = [block for block in final_message.content if block.type == 'tool_use']
            
            if tool_uses:
                # Add the assistant message with tool uses
                transcript.append({
                    "role": "assistant",
                    "content": [block.model_dump() for block in final_message.content]
                })
                
                # Process each tool use
                tool_results = []
                for tool_use in tool_uses:
                    print(f"\n[tool_use:{tool_use.name}] args → {tool_use.input}", flush=True)
                    
                    # Run the tool
                    result = run_tool(tool_use.name, tool_use.input)
                    print(f"[tool_result] {result}", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    })
                
                # Add tool results to transcript
                transcript.append({
                    "role": "user",
                    "content": tool_results
                })
                
                print("\n[continuing with tool results...]\n", flush=True)
                # Continue the conversation with tool results - this should trigger more thinking
                return stream_once(transcript)
            
            return final_message
            
    except Exception as e:
        print(f"\nError in stream: {e}")
        # Fallback to non-streaming with thinking
        response = client.messages.create(
            model=CONFIG["model"],
            max_tokens=CONFIG["max_tokens"],
            tools=TOOLS,
            messages=transcript,
            extra_headers=beta_headers(),
            thinking={"type": "enabled", "budget_tokens": CONFIG["thinking_budget"]},
        )
        return response

# Legacy function removed - no longer needed without mock tools

# ────────────────────────────── 4. Driver ---------------------------------- #
def ask(prompt: str, chat_history: list[dict]):
    # Add user message
    chat_history.append({"role": "user", "content": prompt})
    
    # For weather/search queries, add an explicit tool reminder
    if any(keyword in prompt.lower() for keyword in ['weather', 'temperature', 'restaurant', 'eat', 'food', 'search', 'find']):
        tool_reminder = "REMINDER: Use 'weathertool' for weather queries and 'duckduckgotool' for search/restaurant queries."
        chat_history.append({"role": "user", "content": tool_reminder})
    
    final_msg = stream_once(chat_history)
    
    # Handle both dict and object responses
    if hasattr(final_msg, 'content'):
        content = final_msg.content
    else:
        content = final_msg.get('content', [])
    
    # Extract text content and add to history
    text_parts = []
    for block in content:
        if hasattr(block, 'type') and hasattr(block, 'text'):
            if block.type == 'text':
                text_parts.append(block.text)
        elif isinstance(block, dict) and block.get('type') == 'text':
            text_parts.append(block.get('text', ''))
    
    assistant_response = "".join(text_parts)
    print("\n\n[assistant] " + assistant_response)
    
    # Add assistant response to history
    chat_history.append({"role": "assistant", "content": assistant_response})
    
    return chat_history

def show_config():
    """Display current configuration."""
    print("\n⚙️ Current Configuration:")
    print(f"  Model: {CONFIG['model']}")
    print(f"  Thinking Budget: {CONFIG['thinking_budget']} tokens")
    print(f"  Max Tokens: {CONFIG['max_tokens']}")
    print(f"  Available Models: {', '.join(AVAILABLE_MODELS)}")
    print("\n💡 Use /config model <model_name> to change model")
    print("💡 Use /config thinking <1024-16000> to change thinking budget")

def handle_config_command(args):
    """Handle /config command with subcommands."""
    if not args:
        show_config()
        return
    
    if args[0] == "model":
        if len(args) < 2:
            print("❌ Usage: /config model <model_name>")
            print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
            return
        
        new_model = args[1]
        if new_model not in AVAILABLE_MODELS:
            print(f"❌ Unknown model: {new_model}")
            print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
            return
        
        CONFIG["model"] = new_model
        print(f"✅ Model changed to: {new_model}")
    
    elif args[0] == "thinking":
        if len(args) < 2:
            print("❌ Usage: /config thinking <1024-16000>")
            return
        
        try:
            new_budget = int(args[1])
            if not (1024 <= new_budget <= 16000):
                print("❌ Thinking budget must be between 1024 and 16000")
                return
            
            CONFIG["thinking_budget"] = new_budget
            print(f"✅ Thinking budget changed to: {new_budget} tokens")
        except ValueError:
            print("❌ Thinking budget must be a number")
    
    else:
        print(f"❌ Unknown config option: {args[0]}")
        print("Available options: model, thinking")

def create_tool_awareness_message() -> str:
    """Create a message that informs Claude about available tools."""
    available_tools = get_claude_tools()
    
    if not available_tools:
        return "You currently have no tools available."
    
    # Focus on the key tools for user queries
    key_tools = []
    other_tools = []
    
    for tool in available_tools:
        name = tool['name']
        desc = tool['description'].strip().replace('\n', ' ')[:80]
        
        if 'weather' in name.lower():
            key_tools.append(f"- **{name}**: For weather information and forecasts")
        elif 'duck' in name.lower() or 'search' in name.lower():
            key_tools.append(f"- **{name}**: For internet searches, restaurants, businesses")
        else:
            other_tools.append(f"- **{name}**: {desc}")
    
    tool_sections = []
    if key_tools:
        tool_sections.append("**🌟 KEY TOOLS FOR USER QUERIES:**\n" + "\n".join(key_tools))
    if other_tools:
        tool_sections.append("**🔧 OTHER AVAILABLE TOOLS:**\n" + "\n".join(other_tools))
    
    message = f"""IMPORTANT: You have {len(available_tools)} tools available. 

{chr(10).join(tool_sections)}

CRITICAL: Use these EXACT tool names (e.g., 'weathertool', 'duckduckgotool'). 
When users ask about weather or search for information, USE THE TOOLS ABOVE."""

    return message

def interactive_chat():
    """Interactive chat loop with persistent history."""
    global TOOLS  # Declare at function start
    
    print("🤖 ThinkChain - Claude Chat Interface with Tool Discovery")
    print("Type '/exit', '/quit', or press Ctrl+C to end the conversation.")
    print("Special commands:")
    print("  - /refresh or /reload: Refresh tool discovery")
    print("  - /tools: List all available tools")
    print("  - /config: Show/change configuration (model, thinking budget)")
    print("  - /help: Show this help message\n")
    
    show_config()  # Show initial config
    
    # Initialize chat history with tool awareness
    tool_awareness_msg = create_tool_awareness_message()
    print(f"\n🔧 Initializing with tool awareness for {len(TOOLS)} tools...")
    
    chat_history = [
        {
            "role": "user", 
            "content": tool_awareness_msg
        },
        {
            "role": "assistant",
            "content": "I understand. I now have access to the tools you've listed and will use their exact names when calling them. I'm ready to help you with any tasks that can be accomplished using these tools!"
        }
    ]
    
    try:
        while True:
            user_input = input("\n[you] ").strip()
            
            # Handle slash commands
            if user_input.startswith('/'):
                command_parts = user_input[1:].split()
                command = command_parts[0].lower()
                args = command_parts[1:] if len(command_parts) > 1 else []
                
                if command in ['exit', 'quit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif command in ['refresh', 'reload']:
                    print("🔄 Refreshing tool discovery...")
                    try:
                        tool_count = asyncio.run(refresh_tools())
                        TOOLS = get_claude_tools()
                        
                        # Update Claude's tool awareness
                        tool_awareness_msg = create_tool_awareness_message()
                        chat_history.append({
                            "role": "user",
                            "content": f"🔄 **Tool Discovery Refresh**: {tool_awareness_msg}"
                        })
                        chat_history.append({
                            "role": "assistant", 
                            "content": "Tools refreshed! I'm now aware of the updated tool list and will use these exact tool names."
                        })
                        
                        print(f"✅ Refreshed! Found {tool_count} total tools (local + MCP)")
                        print("🧠 Claude has been updated with the new tool list")
                    except Exception as e:
                        print(f"⚠️  Refresh failed: {e}")
                        print("🔧 Continuing with current tools")
                    continue
                
                elif command == 'tools':
                    print("📋 Available tools:")
                    
                    # Separate local and MCP tools
                    all_tools = list_tools()
                    local_tools = [t for t in all_tools if not t.startswith('mcp_')]
                    mcp_tools = [t for t in all_tools if t.startswith('mcp_')]
                    
                    if local_tools:
                        print(f"Local tools: {', '.join(local_tools)}")
                    else:
                        print("No local tools found")
                    
                    if mcp_tools:
                        print(f"MCP tools: {', '.join(mcp_tools)}")
                    else:
                        print("No MCP tools found")
                    
                    print(f"Total: {len(TOOLS)} tools")
                    continue
                
                elif command == 'config':
                    handle_config_command(args)
                    continue
                
                elif command == 'help':
                    print("🤖 ThinkChain - Claude Chat Interface with Tool Discovery")
                    print("Special commands:")
                    print("  - /refresh or /reload: Refresh tool discovery")
                    print("  - /tools: List all available tools")
                    print("  - /config: Show/change configuration")
                    print("    - /config model <model_name>: Change model")
                    print("    - /config thinking <1024-16000>: Change thinking budget")
                    print("  - /help: Show this help message")
                    print("  - /exit or /quit: End conversation")
                    continue
                
                else:
                    print(f"❌ Unknown command: /{command}")
                    print("Type /help for available commands")
                    continue
            
            # Handle legacy commands (without slash) for backward compatibility
            elif user_input.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye! (Tip: use /exit for slash commands)")
                break
            
            elif not user_input:
                continue
            
            # Handle legacy commands
            elif user_input.lower() in ['refresh', 'reload']:
                print("🔄 Refreshing tool discovery... (Tip: use /refresh for slash commands)")
                try:
                    tool_count = asyncio.run(refresh_tools())
                    TOOLS = get_claude_tools()
                    
                    # Update Claude's tool awareness
                    tool_awareness_msg = create_tool_awareness_message()
                    chat_history.append({
                        "role": "user",
                        "content": f"🔄 **Tool Discovery Refresh**: {tool_awareness_msg}"
                    })
                    chat_history.append({
                        "role": "assistant", 
                        "content": "Tools refreshed! I'm now aware of the updated tool list and will use these exact tool names."
                    })
                    
                    print(f"✅ Refreshed! Found {tool_count} total tools (local + MCP)")
                    print("🧠 Claude has been updated with the new tool list")
                except Exception as e:
                    print(f"⚠️  Refresh failed: {e}")
                    print("🔧 Continuing with current tools")
                continue
            
            elif user_input.lower() in ['list tools', 'tools']:
                print("📋 Available tools... (Tip: use /tools for slash commands)")
                
                # Separate local and MCP tools
                all_tools = list_tools()
                local_tools = [t for t in all_tools if not t.startswith('mcp_')]
                mcp_tools = [t for t in all_tools if t.startswith('mcp_')]
                
                if local_tools:
                    print(f"Local tools: {', '.join(local_tools)}")
                else:
                    print("No local tools found")
                
                if mcp_tools:
                    print(f"MCP tools: {', '.join(mcp_tools)}")
                else:
                    print("No MCP tools found")
                
                print(f"Total: {len(TOOLS)} tools")
                continue
            
            elif user_input.lower() == 'help':
                print("🤖 ThinkChain - Claude Chat Interface with Tool Discovery (Legacy)")
                print("Special commands (legacy - use /command for new format):")
                print("  - refresh or reload: Refresh tool discovery")
                print("  - list tools or tools: List all available tools")
                print("  - help: Show this help message")
                print("  - exit or quit: End conversation")
                print("\n💡 Try the new slash commands: /help, /tools, /config, etc.")
                continue
            
            # Process as regular chat message
            else:
                chat_history = ask(user_input, chat_history)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_chat()
