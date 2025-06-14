#!/usr/bin/env python3
# /// script
# dependencies = [
#     "anthropic>=0.25.0",
#     "sseclient-py",
#     "pydantic",
#     "python-dotenv",
#     "rich>=13.0.0",
#     "prompt-toolkit>=3.0.0",
#     "click>=8.0.0",
#     "yaspin>=3.0.0",
#     "requests",
#     "beautifulsoup4",
#     "mcp",
#     "httpx",
# ]
# ///

"""
ThinkChain Enhanced UI

Interactive CLI experience with rich formatting, progress bars, and beautiful display
while preserving all the original functionality including streaming and thinking capabilities.
"""

import os, json, itertools, time, asyncio
from collections import defaultdict

import anthropic              # ≥0.25.0
from sseclient import SSEClient  # pip install sseclient‑py
from dotenv import load_dotenv

# Import our tool discovery system
from tool_discovery import (
    get_tool_discovery, get_claude_tools, execute_tool_sync, 
    refresh_tools, list_tools, initialize_tool_discovery
)

# Import enhanced UI components
from ui_components import ui, print_banner, print_initialization_progress, format_command_suggestions

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

# Initialize tool discovery with enhanced UI feedback
def initialize_tools_with_progress():
    """Initialize tools with beautiful progress display"""
    steps = [
        "🔍 Discovering local tools...",
        "🌐 Initializing MCP integration...", 
        "🔧 Building tool registry...",
        "✅ Finalizing setup..."
    ]
    
    print_initialization_progress(steps)
    
    try:
        ui.print("🔍 Discovering local tools...")
        tool_discovery = get_tool_discovery()
        local_count = len(tool_discovery.discovered_tools)
        
        ui.print("🌐 Initializing MCP integration...")
        try:
            total_tools = asyncio.run(initialize_tool_discovery())
            mcp_count = total_tools - local_count
            ui.print_success(f"Initialized {total_tools} total tools ({local_count} local + {mcp_count} MCP)")
        except Exception as e:
            ui.print_warning(f"MCP initialization failed: {e}")
            ui.print("🔧 Continuing with local tools only")
            total_tools = local_count
            mcp_count = 0
        
        # Get all available tools (local + MCP)
        tools = get_claude_tools()
        
        return tools, local_count, mcp_count
        
    except Exception as e:
        ui.print_error("Tool initialization failed", str(e))
        return [], 0, 0

# Initialize tools
TOOLS, LOCAL_TOOL_COUNT, MCP_SERVER_COUNT = initialize_tools_with_progress()

# ────────────────────────────── 2. Helpers ───────────────────────────── #
def beta_headers() -> dict[str, str]:
    """Return extra headers for every call."""
    return {"anthropic-beta": BETA_HEADERS}

def run_tool(name: str, args: dict) -> str:
    """Dispatch to discovered tool function and stringify the result."""
    start_time = time.time()
    
    # Show tool execution start
    ui.print_tool_execution(name, "executing")
    
    # Execute discovered tool (using sync wrapper)
    try:
        result = execute_tool_sync(name, args)
        duration = time.time() - start_time
        ui.print_tool_execution(name, "completed", duration)
        return result
    except Exception as exc:
        duration = time.time() - start_time
        ui.print_tool_execution(name, "failed", duration)
        return f"<error>Tool {name} execution failed: {exc}</error>"

# ────────────────────────────── 3. Streaming loop ────────────────────── #
def stream_once(transcript: list[dict]) -> dict:
    """
    Send one request, handle tools if any, and return the final message.
    Shows thinking process including after tool execution with enhanced UI.
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
                            ui.print(f"\n🔧 [bold yellow]Tool Use:[/bold yellow] [cyan]{chunk.content_block.name}[/cyan]")
                        elif chunk.content_block.type == 'thinking':
                            ui.print(f"\n💭 [bold blue]Thinking:[/bold blue] ", end='')
                            in_thinking = True
                elif chunk.type == 'content_block_delta':
                    if hasattr(chunk.delta, 'text') and chunk.delta.text:
                        if in_thinking:
                            ui.console.print(chunk.delta.text, end='', style="italic blue")
                            thinking_content.append(chunk.delta.text)
                        else:
                            ui.console.print(chunk.delta.text, end='')
                elif chunk.type == 'content_block_stop':
                    if in_thinking:
                        ui.print("")  # New line after thinking
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
                    ui.print(f"\n🔧 [bold cyan]Executing:[/bold cyan] {tool_use.name}")
                    ui.print_json(tool_use.input, f"Arguments for {tool_use.name}")
                    
                    # Run the tool
                    result = run_tool(tool_use.name, tool_use.input)
                    
                    # Format result nicely
                    try:
                        # Try to parse as JSON for better display
                        parsed_result = json.loads(result)
                        ui.print_json(parsed_result, f"Result from {tool_use.name}")
                    except:
                        # If not JSON, just show as text
                        ui.print_panel(result, f"Result from {tool_use.name}", "success")
                    
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
                
                ui.print("\n🔄 [bold blue]Continuing with tool results...[/bold blue]\n")
                # Continue the conversation with tool results - this should trigger more thinking
                return stream_once(transcript)
            
            return final_message
            
    except Exception as e:
        ui.print_error("Stream error", str(e))
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

# ────────────────────────────── 4. Enhanced Commands ──────────────────── #
def show_tools_command():
    """Enhanced tools listing command"""
    ui.print_rule("Available Tools")
    
    # Categorize tools
    tools_by_category = {
        'local': [],
        'mcp': []
    }
    
    # Get discovered tools
    all_tools = list_tools()
    local_tools = [t for t in all_tools if not t.startswith('mcp_')]
    mcp_tools = [t for t in all_tools if t.startswith('mcp_')]
    
    # Add local tools
    for tool_name in local_tools:
        tools_by_category['local'].append({
            'name': tool_name,
            'description': f"Local tool: {tool_name}",
            'type': 'Local'
        })
    
    # Add MCP tools
    for tool_name in mcp_tools:
        tools_by_category['mcp'].append({
            'name': tool_name,
            'description': f"MCP tool: {tool_name.replace('mcp_', '')}",
            'type': 'MCP'
        })
    
    ui.print_tool_table(tools_by_category)
    ui.print(f"\n📊 [bold]Total:[/bold] {len(TOOLS)} tools available")

def show_help_command():
    """Enhanced help command"""
    commands = {
        "/help": "Show this help message",
        "/tools": "List all available tools in detail",
        "/refresh": "Refresh tool discovery (local + MCP)",
        "/status": "Show system status and configuration",
        "/clear": "Clear the screen",
        "/config": "Show/edit configuration (model, thinking budget)",
        "/config model <name>": "Change Claude model (sonnet/opus)",
        "/config thinking <1024-16000>": "Change thinking token budget",
        "/exit": "Exit the application"
    }
    
    ui.print_help(commands)
    
    ui.print_panel(
        "💡 [bold]Tips:[/bold]\n"
        "• Use Tab for command completion (works with / commands)\n"
        "• Use ↑/↓ arrows for command history\n"
        "• Type '/tools' to see all available tools\n"
        "• Legacy commands (without /) still work for backward compatibility\n"
        "• MCP tools are prefixed with 'mcp_'", 
        "Getting Started", 
        "info"
    )

def show_status_command():
    """Show detailed system status"""
    ui.print_status_bar(LOCAL_TOOL_COUNT, MCP_SERVER_COUNT, True, "Ready")
    
    # Additional status info
    status_info = f"""
🔧 **Local Tools:** {LOCAL_TOOL_COUNT} discovered
🌐 **MCP Servers:** {MCP_SERVER_COUNT} configured  
💭 **Thinking:** Enabled ({CONFIG['thinking_budget']} token budget)
🤖 **Model:** {CONFIG['model']}
🔋 **Status:** Ready for interactions
    """
    
    ui.print_markdown(status_info)

def show_config_command():
    """Enhanced config display command"""
    config_data = {
        "Model": CONFIG['model'],
        "Thinking Budget": f"{CONFIG['thinking_budget']} tokens",
        "Max Tokens": CONFIG['max_tokens'],
        "Available Models": ", ".join(AVAILABLE_MODELS)
    }
    
    ui.print_panel(
        "\n".join([f"{k}: {v}" for k, v in config_data.items()]) + 
        "\n\n💡 Use /config model <model_name> to change model" +
        "\n💡 Use /config thinking <1024-16000> to change thinking budget",
        "Configuration",
        "info"
    )

def handle_config_command(args):
    """Handle enhanced /config command with subcommands."""
    if not args:
        show_config_command()
        return
    
    if args[0] == "model":
        if len(args) < 2:
            ui.print_error("Usage", "/config model <model_name>")
            ui.print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
            return
        
        new_model = args[1]
        if new_model not in AVAILABLE_MODELS:
            ui.print_error("Unknown model", new_model)
            ui.print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
            return
        
        CONFIG["model"] = new_model
        ui.print_success(f"Model changed to: {new_model}")
    
    elif args[0] == "thinking":
        if len(args) < 2:
            ui.print_error("Usage", "/config thinking <1024-16000>")
            return
        
        try:
            new_budget = int(args[1])
            if not (1024 <= new_budget <= 16000):
                ui.print_error("Invalid range", "Thinking budget must be between 1024 and 16000")
                return
            
            CONFIG["thinking_budget"] = new_budget
            ui.print_success(f"Thinking budget changed to: {new_budget} tokens")
        except ValueError:
            ui.print_error("Invalid input", "Thinking budget must be a number")
    
    else:
        ui.print_error("Unknown config option", args[0])
        ui.print("Available options: model, thinking")

def refresh_command(chat_history):
    """Enhanced refresh command with progress"""
    ui.print("🔄 Refreshing tool discovery...")
    
    with ui.progress_context("Refreshing tools..."):
        try:
            tool_count = asyncio.run(refresh_tools())
            global TOOLS, LOCAL_TOOL_COUNT, MCP_SERVER_COUNT
            
            # Update tool counts
            tool_discovery = get_tool_discovery()
            LOCAL_TOOL_COUNT = len(tool_discovery.discovered_tools)
            MCP_SERVER_COUNT = tool_count - LOCAL_TOOL_COUNT if tool_count > LOCAL_TOOL_COUNT else 0
            
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
            
            ui.print_success(f"Refreshed! Found {tool_count} total tools ({LOCAL_TOOL_COUNT} local + {MCP_SERVER_COUNT} MCP)")
            ui.print("🧠 Claude has been updated with the new tool list")
            
        except Exception as e:
            ui.print_error("Refresh failed", str(e))
            ui.print("🔧 Continuing with current tools")

# ────────────────────────────── 5. Main Chat Loop ──────────────────────── #
def ask(prompt: str, chat_history: list[dict]):
    """Process user input and get Claude's response with enhanced formatting"""
    # Add user message
    chat_history.append({"role": "user", "content": prompt})
    
    # For weather/search queries, add an explicit tool reminder
    if any(keyword in prompt.lower() for keyword in ['weather', 'temperature', 'restaurant', 'eat', 'food', 'search', 'find']):
        tool_reminder = "REMINDER: Use 'weathertool' for weather queries and 'duckduckgotool' for search/restaurant queries."
        chat_history.append({"role": "user", "content": tool_reminder})
        ui.print("🔧 [dim]Added tool reminder for Claude[/dim]")
    
    ui.print(f"\n👤 [bold blue]You:[/bold blue] {prompt}")
    
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
    ui.print_claude_response(assistant_response)
    
    # Add assistant response to history
    chat_history.append({"role": "assistant", "content": assistant_response})
    
    return chat_history

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
    """Enhanced interactive chat loop with rich UI"""
    # Clear screen and show banner
    ui.clear_screen()
    print_banner()
    
    # Show initial status
    ui.print_status_bar(LOCAL_TOOL_COUNT, MCP_SERVER_COUNT, True, "Ready")
    
    ui.print_panel(
        "Welcome to ThinkChain! 🚀\n\n"
        "I have access to multiple tools for various tasks. Use slash commands like /help, /tools, /config\n"
        "or just start chatting and I'll use tools as needed to help you.\n\n"
        "✨ Enhanced features: Thinking visualization, tool execution tracking, and rich formatting!",
        "Welcome",
        "primary"
    )
    
    # Show initial config
    show_config_command()
    
    # Initialize chat history with tool awareness
    chat_history = [
        {
            "role": "user", 
            "content": create_tool_awareness_message()
        },
        {
            "role": "assistant",
            "content": "I understand. I now have access to the tools you've listed and will use their exact names when calling them. I'm ready to help you with any tasks that can be accomplished using these tools!"
        }
    ]
    
    # Available commands for autocomplete (with slash prefixes)
    command_words = ['/help', '/tools', '/refresh', '/reload', '/status', '/clear', '/config', '/exit', '/quit',
                    'help', 'tools', 'refresh', 'reload', 'status', 'clear', 'debug', 'config', 'exit', 'quit']
    
    try:
        while True:
            try:
                # Get user input with autocomplete
                user_input = ui.get_input("Enter command or message", command_words).strip()
                
                # Handle slash commands
                if user_input.startswith('/'):
                    command_parts = user_input[1:].split()
                    command = command_parts[0].lower()
                    args = command_parts[1:] if len(command_parts) > 1 else []
                    
                    if command in ['exit', 'quit']:
                        ui.print_success("👋 Goodbye!")
                        break
                    
                    elif command in ['refresh', 'reload']:
                        refresh_command(chat_history)
                        continue
                    
                    elif command == 'tools':
                        show_tools_command()
                        continue
                    
                    elif command == 'config':
                        handle_config_command(args)
                        continue
                    
                    elif command == 'help':
                        show_help_command()
                        continue
                    
                    elif command == 'status':
                        show_status_command()
                        continue
                    
                    elif command == 'clear':
                        ui.clear_screen()
                        print_banner()
                        ui.print_status_bar(LOCAL_TOOL_COUNT, MCP_SERVER_COUNT, True, "Ready")
                        continue
                    
                    else:
                        ui.print_error("Unknown command", f"/{command}")
                        ui.print("Type /help for available commands")
                        continue
                
                # Handle legacy commands (backward compatibility)
                elif user_input.lower() in ['exit', 'quit']:
                    ui.print_success("👋 Goodbye! (Tip: use /exit for slash commands)")
                    break
                
                elif not user_input:
                    continue
                
                elif user_input.lower() in ['refresh', 'reload']:
                    ui.print("🔄 Refreshing... (Tip: use /refresh for slash commands)")
                    refresh_command(chat_history)
                    continue
                
                elif user_input.lower() in ['list tools', 'tools']:
                    ui.print("📋 Showing tools... (Tip: use /tools for slash commands)")
                    show_tools_command()
                    continue
                
                elif user_input.lower() == 'help':
                    ui.print("📖 Showing help... (Tip: use /help for slash commands)")
                    show_help_command()
                    continue
                
                elif user_input.lower() == 'status':
                    ui.print("📊 Showing status... (Tip: use /status for slash commands)")
                    show_status_command()
                    continue
                
                elif user_input.lower() == 'clear':
                    ui.print("🧹 Clearing... (Tip: use /clear for slash commands)")
                    ui.clear_screen()
                    print_banner()
                    ui.print_status_bar(LOCAL_TOOL_COUNT, MCP_SERVER_COUNT, True, "Ready")
                    continue
                
                # Process as regular chat message
                else:
                    chat_history = ask(user_input, chat_history)
                
            except KeyboardInterrupt:
                ui.print("\n\n👋 Goodbye!")
                break
                
    except Exception as e:
        ui.print_error("Unexpected error in chat loop", str(e))

if __name__ == "__main__":
    interactive_chat()