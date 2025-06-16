ThinkChain: When Claude's Thinking Meets Tool Feedback Loops
Martin Bowling Jun 14, 2025

----------------

The Inspiration!
I saw Pietro Schirano's tweet about his "chain of tools" and immediately thought - I must build this! You see, I've been obsessed with Claude's tool use capabilities lately, especially after Anthropic released their interleaved thinking features. Most Claude integrations I'd seen treated tools as black boxes - call a tool, get a result, move on. But what if tool results could feed back into Claude's thinking process in real-time?

That simple question led me down a rabbit hole that resulted in ThinkChain - a system where thinking, tool execution, and reasoning form a continuous feedback loop. Instead of the traditional linear flow of "call tool → get result → respond," ThinkChain creates something much more powerful: "think → call tool → think about results → respond intelligently."

What I discovered surprised me. When you inject tool results back into Claude's thinking stream, it doesn't just use tools - it becomes dramatically smarter about how it uses them. Here's what I built, what I learned, and why this changes everything about AI tool integration.

The Core Innovation: Tool Result Injection
Let me show you the difference with a real example. Ask a traditional Claude integration "What's the weather in San Francisco and where should I eat dinner there?" and you get this flow:

Traditional approach:

User Question → Claude thinks → Calls weather tool → Gets result
               → Calls restaurant tool → Gets result → Combines results
ThinkChain approach:


User Question → Claude thinks → Calls weather tool → Thinks about weather
               → Calls restaurant tool with weather context → Thinks about both
               → Synthesizes intelligent response


The magic happens in those thinking steps after tool execution. Here's the actual technical implementation that makes this possible:

```python
async def stream_once(messages, tools):
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        messages=messages,
        tools=tools,
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"},
        thinking={"type": "enabled", "budget_tokens": 1024}
    ) as stream:

        async for event in stream:
            if event.type == "tool_use":
                # Execute the tool
                result = await execute_tool(event.name, event.input)

                # This is the key: inject result back into thinking stream
                transcript.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "content": result}]
                })

                # Continue streaming - Claude thinks about the results
                return await stream_once(transcript)
```

This creates a feedback loop where Claude's initial thinking leads to tool use, tool results inform continued thinking, and the final response incorporates both reasoning and tool outcomes. It's not just smarter - it's thinking smarter.

Architecture Deep Dive: How It All Works
Building ThinkChain taught me that the real power isn't in having lots of tools - it's in how tools discover each other, execute cleanly, and feed results back intelligently. Here's how I architected it:

The Tool Discovery System
I wanted developers to just drop a Python file in a folder and have it work. No registration, no complex setup. Here's the discovery pipeline:

```
Local Tools (/tools/*.py) → Validation → Registry
                                         ↓
MCP Servers (config.json) → Connection → Registry → Unified Tool List → Claude API
```

Every tool implements this simple interface:

```python
from tools.base import BaseTool

class WeatherTool(BaseTool):
    name = "weathertool"
    description = """
    Gets current weather information for any location worldwide.

    Use this tool when users ask about:
    - Current weather in any city/location
    - Temperature anywhere
    - "What's the weather like in [location]?"
    """

    input_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state/country"},
            "units": {"type": "string", "enum": ["fahrenheit", "celsius"], "default": "fahrenheit"}
        },
        "required": ["location"]
    }

    def execute(self, **kwargs) -> str:
        location = kwargs.get("location")
        # Hit wttr.in API for real weather data
        response = requests.get(f"https://wttr.in/{location}?format=j1")
        data = response.json()

        # Format for Claude
        return f"🌤️ Weather for {location}:\nTemperature: {data['current_condition'][0]['temp_F']}°F\n..."
```

The beauty is that tools are just classes with four things: a name, description, input schema, and execute method. Drop the file in /tools/, and ThinkChain discovers it automatically.

Real Example Flow
Let me show you what happens when someone asks "What's the weather in San Francisco and find good restaurants there?":


```bash
[thinking] I need to check the weather first, then find restaurants that might be good for those conditions.

[tool_use:weathertool] ▶ {"location": "San Francisco, CA"}
[tool_result] 🌤️ Weather for San Francisco, CA:
Temperature: 62°F (feels like 62°F)
Conditions: Partly cloudy
Humidity: 38%
Wind: 5 mph WSW

[thinking] It's a pleasant 62°F and partly cloudy - perfect weather for outdoor dining or walking to restaurants. I should look for places with outdoor seating or patios.

[tool_use:duckduckgotool] ▶ {"query": "best restaurants San Francisco outdoor seating patio"}
[tool_result] [Restaurant results with outdoor dining options...]

[thinking] Given the nice weather, I can recommend these outdoor-friendly restaurants...
```

See how the weather result influences the restaurant search? That's the power of tool result injection - Claude doesn't just call tools sequentially, it thinks about results and makes smarter decisions.

Building Real Tools: From Concept to Code
When I started building tools for ThinkChain, I learned that the description is just as important as the implementation. Claude needs to understand not just what your tool does, but when to use it.

Here's the complete weathertool implementation with everything I learned:

```python
from tools.base import BaseTool
import requests
import json

class WeatherTool(BaseTool):
    name = "weathertool"

    # This description is crucial - it helps Claude decide when to use the tool
    description = """
    Gets current weather information for any location worldwide. Returns temperature, 
    weather conditions, humidity, wind speed and direction.

    Use this tool when users ask about:
    - Current weather in any city/location
    - Temperature anywhere  
    - Weather conditions (sunny, cloudy, rainy, etc.)
    - "What's the weather like in [location]?"
    """

    # JSON Schema for input validation
    input_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state/country (e.g., 'San Francisco, CA' or 'London, UK')"
            },
            "units": {
                "type": "string", 
                "description": "Temperature units",
                "enum": ["fahrenheit", "celsius", "kelvin"],
                "default": "fahrenheit"
            }
        },
        "required": ["location"]
    }

    def execute(self, **kwargs) -> str:
        location = kwargs.get("location")
        units = kwargs.get("units", "fahrenheit")

        try:
            # Use wttr.in - free weather API, no key needed
            response = requests.get(f"https://wttr.in/{location}?format=j1", timeout=10)
            response.raise_for_status()
            data = response.json()

            current = data['current_condition'][0]
            temp_c = int(current['temp_C'])
            temp_f = int(current['temp_F'])

            # Format based on requested units
            if units.lower() == "celsius":
                temp = f"{temp_c}°C"
            else:  # Default to fahrenheit
                temp = f"{temp_f}°F"

            # Return formatted result that Claude can easily understand
            return f"""🌤️ Weather for {location}:
Temperature: {temp}
Conditions: {current['weatherDesc'][0]['value']}
Humidity: {current['humidity']}%
Wind: {current['windspeedMiles']} mph {current['winddir16Point']}"""

        except Exception as e:
            # Always return string errors - Claude can handle them gracefully
            return f"❌ Error fetching weather data: {str(e)}"
```

Key Patterns I Discovered
Rich Descriptions Win: The more context you give Claude about when to use your tool, the better it performs. Include example queries, keywords that should trigger it, and specific use cases.

Error Handling Matters: Always catch exceptions and return string error messages. Claude is surprisingly good at handling errors gracefully when you give it clear information about what went wrong.

Format for Claude: Structure your output to be easily parseable. Use emojis, clear labels, and consistent formatting. Claude works better with well-structured data.

Input Validation: Use comprehensive JSON schemas. They prevent errors and help Claude understand exactly what parameters your tool expects.

MCP Integration: Extending Beyond Local Tools
One of the most exciting discoveries was integrating with MCP (Model Context Protocol) servers. MCP lets you connect to external servers that provide tools, dramatically expanding what's possible.

Here's how I added SQLite database operations:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./database.db"],
      "description": "SQLite database operations",
      "enabled": true
    }
  }
}
```

Just by adding that configuration and running uvx install mcp-server-sqlite, ThinkChain gained 6 new tools:

```python
mcp_sqlite_read_query # Execute SELECT queries

mcp_sqlite_write_query # Execute INSERT/UPDATE/DELETE

mcp_sqlite_create_table # Create database tables

mcp_sqlite_list_tables # List all tables

mcp_sqlite_describe_table # Get table schema

mcp_sqlite_append_insight # Add business insights
```

The power comes from combining ecosystems. Now I can ask: "Check the weather in our office locations from the database, then find restaurants near each one" and Claude seamlessly uses both local tools (weather) and MCP tools (database) together.

What blew my mind was how naturally Claude chains these together. It doesn't see a difference between local Python tools and remote MCP servers - they're all just tools in its toolkit.

The Enhanced UI: Making It Beautiful
Here's something I learned early: if you're building developer tools, the experience matters just as much as the functionality. I could have stopped at a basic CLI, but I wanted ThinkChain to feel as intelligent as it actually is.

So I built two interfaces:

`thinkchain.py` - The full experience with Rich formatting, progress bars, and interactive features `thinkchain_cli.py` - Minimal CLI for when you just need it to work run.py - Smart launcher that detects available libraries and picks the best option

Here's what the enhanced UI looks like when it starts up:


```bash
╔═══════════════════════════════════════════════════════════════════╗
║  ████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗ ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗  ║
║  ╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝██╔════╝██║  ██║██╔══██╗██║████╗  ██║  ║
║     ██║   ███████║██║██╔██╗ ██║█████╔╝ ██║     ███████║███████║██║██╔██╗ ██║  ║
║     ██║   ██╔══██║██║██║╚██╗██║██╔═██╗ ██║     ██╔══██║██╔══██║██║██║╚██╗██║  ║
║     ██║   ██║  ██║██║██║ ╚████║██║  ██╗╚██████╗██║  ██║██║  ██║██║██║ ╚████║  ║
║     ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝  ║
║          🧠 Claude Chat with Advanced Tool Integration & Thinking 💭          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Claude Tool Discovery Chat
🔧 Local: 11 tools │ 🌐 MCP: 6 servers │ 💭 Thinking: ON │ 🔋 Ready
```

But the real magic happens during conversations. Watch what happens when Claude uses a tool:


```bash
👤 You: What's the weather in Cross Lanes, WV?

💭 Thinking: I'll check the current weather in Cross Lanes, WV for you.

🔧 Tool Use: weathertool

🔧 Executing: weathertool
╭───────────────────────── Arguments for weathertool ──────────────────────────╮
│ {                                                                            │
│   "location": "Cross Lanes, WV"                                              │
│ }                                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
🔧 weathertool: Executing...
🔧 weathertool: Completed (0.8s)
╭────────────────────────── Result from weathertool ───────────────────────────╮
│ 🌤️ Weather for Cross Lanes, WV:                                               │
│ Temperature: 73°F (feels like 77°F)                                          │
│ Conditions: Heavy rain with thunderstorm                                     │
│ Humidity: 79%                                                                │
│ Wind: 2 mph WSW                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

🔄 Continuing with tool results...

💭 Thinking: The current weather in Cross Lanes, WV shows stormy conditions...

🤖 Claude: The current weather in Cross Lanes, WV is stormy with heavy rain and thunderstorms...
```

Every step is visualized: thinking appears in italic blue, tool execution shows progress with timing, and results are formatted in beautiful boxes. You can actually watch Claude think through problems.

Technical Implementation
The enhanced UI is built with the Rich library, but here's the clever part - it gracefully degrades:

```python
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress
    from ui_components import ui  # Enhanced UI components
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False

def print_tool_execution(name, status, duration=None):
    if UI_AVAILABLE:
        if status == "executing":
            ui.print(f"🔧 [yellow]Executing:[/yellow] {name}")
        elif status == "completed":
            ui.print(f"🔧 [green]Completed:[/green] {name} ({duration:.1f}s)")
    else:
        # Fallback to basic text
        print(f"[tool_use:{name}] {status}")
```

The slash commands system was particularly fun to build:

```python
command_words = ['/help', '/tools', '/refresh', '/config', '/exit']

user_input = ui.get_input("Enter command or message", command_words)

if user_input.startswith('/'):
    command_parts = user_input[1:].split()
    command = command_parts[0].lower()

    if command == 'tools':
        show_tools_command()  # Beautiful table of all tools
    elif command == 'config':
        handle_config_command(args)  # Interactive configuration
```

You get tab completion, command history with arrow keys, and rich formatting throughout. But if you don't have Rich installed, everything still works - it just falls back to plain text.

Lessons Learned and Developer Insights
Building ThinkChain taught me things about AI tool integration that I never expected. Here are the biggest insights:

What Worked Incredibly Well
Tool result injection is a game-changer. I cannot overstate this. When Claude can think about tool results before responding, the quality of responses improves dramatically. It's not just using tools - it's reasoning about their outputs.

Automatic tool discovery scales effortlessly. I started with 2 tools, now have 17, and adding new ones is still just "drop file in folder, restart." The discovery system handles all the complexity.

Rich descriptions make Claude smarter. The difference between a tool with a basic description and one with rich context about when to use it is night and day. Claude makes much better tool selection decisions with good descriptions.

MCP integration unlocks unlimited possibilities. Once I connected to MCP servers, I realized this isn't just about the tools I build - it's about connecting to an entire ecosystem.

Challenges That Surprised Me
Managing async MCP connections was trickier than expected. MCP servers run as separate processes, and coordinating their lifecycle with the main application required careful async handling:

```python
async def cleanup_mcp_servers():
    """Gracefully shutdown all MCP server connections"""
    for server_name, client in self.active_clients.items():
        try:
            await client.close()
        except Exception as e:
            logger.error(f"Error during cleanup of MCP server {server_name}: {e}")
```

Tool failure handling needs to be bulletproof. When a tool fails, you can't just crash - Claude needs to understand what went wrong and potentially try alternative approaches:

```python
def execute_tool_sync(name: str, args: dict) -> str:
    try:
        result = tool_function(args)
        return result
    except requests.RequestException as e:
        return f"❌ Network error calling {name}: {str(e)}"
    except ValidationError as e:
        return f"❌ Invalid input for {name}: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error in {name}: {str(e)}"
```

Thinking budget optimization matters more than I thought. Initially I set the thinking budget to 16,000 tokens, but I found that 1,024-2,048 tokens often works better. Too much thinking budget and Claude overthinks simple problems. Too little and it can't reason through complex tool chains.

Performance Insights
Tool execution timing varies wildly. Weather API calls take 0.5-1 second, web scraping can take 3-5 seconds, and database operations are nearly instant. The UI progress indicators help users understand what's happening.

Streaming vs. batch processing trade-offs. Streaming gives better user experience but requires more complex error handling. I ended up with a hybrid approach - streaming for the conversation, but batch processing for tool discovery at startup.

Memory usage grows with tool count. Each tool keeps its schema in memory, and MCP connections maintain persistent state. With 17 tools I'm using about 50MB, which is totally reasonable, but it's something to watch.

Code Patterns That Emerged
Here are some patterns I found myself using repeatedly:


Copy
```python
# Tool discovery pattern
def discover_tools():
    tool_schemas = []
    for tool_file in os.listdir(TOOLS_DIR):
        if tool_file.endswith('.py'):
            tool_module = importlib.import_module(f"tools.{tool_file[:-3]}")
            tool_schemas.append(tool_module.tool_schema)
    return tool_schemas


# Tool result validation pattern
def validate_and_format_result(result: str, tool_name: str) -> str:
    if not result:
        return f"❌ {tool_name} returned empty result"

    # Try to parse as JSON for structured data
    try:
        parsed = json.loads(result)
        return json.dumps(parsed, indent=2)  # Pretty print
    except:
        return result  # Return as-is if not JSON

# Graceful degradation pattern  
def safe_tool_execution(tool_func, *args, **kwargs):
    try:
        return tool_func(*args, **kwargs)
    except ImportError as e:
        return f"❌ Missing dependency: {e}"
    except Exception as e:
        return f"❌ Tool execution failed: {e}"

# Configuration management pattern
def update_config(key: str, value: Any) -> bool:
    if key in ALLOWED_CONFIG_KEYS:
        CONFIG[key] = value
        save_config_to_file()  # Persist changes
        return True
    return False
```

What I realized is that building AI tools isn't just about the AI part - it's about creating robust, developer-friendly systems that handle edge cases gracefully and provide great experiences.

Fork It and Make It Yours
Here's the thing - ThinkChain is designed to be forked and extended. I built it with MIT license specifically because I want to see what developers build with it.

The architecture is modular by design. Want to add tools for your domain? Drop Python files in /tools/. Want to connect to specialized MCP servers? Edit mcp_config.json. Want to customize the UI? Modify the Rich components.

Ideas for Domain-Specific Forks
Data Science ThinkChain: Add pandas tools for data manipulation, matplotlib for visualization, jupyter tools for notebook integration. Imagine asking Claude to "load this dataset, analyze trends, and create visualizations" and watching it think through each step.

Web Development ThinkChain: React component generators, npm package managers, git integration tools, deployment automation. "Create a new React component with these props and add it to the project" becomes a conversation, not a manual process.

DevOps ThinkChain: Docker container tools, Kubernetes deployment tools, AWS/GCP integration, monitoring dashboards. "Check the health of our production services and scale if needed" with full reasoning about the decisions.

Research ThinkChain: Academic paper search tools, citation managers, data analysis tools, LaTeX generators. "Find recent papers on this topic and summarize their methodologies" with tool-driven research.

Getting Started with Your Fork
The process is straightforward:

```bash
# Fork and clone
git clone https://github.com/yourusername/your-thinkchain-fork.git
cd your-thinkchain-fork

# Install dependencies
uv pip install -r requirements.txt

# Create your first tool
vim tools/yourtool.py

# Test it
python thinkchain.py
/refresh  # Loads your new tool
"Use my new tool for X"  # Test with Claude
```

What I Hope You Build
I'm excited to see domain-specific forks, novel tool combinations, and creative MCP integrations. Maybe someone builds ThinkChain for legal research, or scientific computing, or creative writing. The possibilities are endless.

If you build something cool, let me know! I'd love to feature community forks and see how people extend the system.

What's Next
Building ThinkChain opened my eyes to what's possible when AI tools can think about their own tool use. Here's what I'm excited about for the future:

Technical Improvements I'm Working On
Better error recovery: When tools fail, Claude should be able to suggest alternative approaches or debug the problem. I'm experimenting with giving Claude access to error logs and system state.

Tool composition workflows: Instead of just chaining tools, what if Claude could compose them into reusable workflows? "Remember this sequence of tools as a 'data analysis workflow' for future use."

Multi-model support: Claude is amazing, but different models have different strengths. What if you could use GPT-4 for creative tasks and Claude for analytical ones, all in the same conversation?

Performance optimizations: Some tool chains could run in parallel instead of sequentially. I'm exploring how to let Claude mark which tools can run concurrently.

The Bigger Picture
What excites me most is that ThinkChain represents a shift from "AI that uses tools" to "AI that thinks about tools." When Claude can reason about tool results, it makes fundamentally better decisions about which tools to use and how to use them.

I think this is just the beginning. As more MCP servers come online, as tool ecosystems mature, and as AI models get better at reasoning, we're going to see AI systems that don't just automate tasks - they intelligently orchestrate complex workflows.

The future isn't AI replacing human developers - it's AI becoming incredibly sophisticated development partners that can think through problems, use tools intelligently, and explain their reasoning every step of the way.

Conclusion
Pietro's tweet about "chain of tools" sparked an idea, but what I discovered while building ThinkChain was something bigger: when you let AI think about tool results, everything changes.

Claude doesn't just use tools anymore - it reasons about them, learns from them, and makes intelligent decisions about how to combine them. The feedback loop between thinking and tool execution creates a kind of intelligence I hadn't seen before.

For developers, this means we need to think differently about AI integration. It's not enough to just give AI access to tools - we need to design systems that let AI think about tool results and use that thinking to make better decisions.

The technical patterns are surprisingly straightforward: tool result injection, async streaming, graceful error handling, and rich user experiences. But the implications are profound. We're moving from AI assistants that follow scripts to AI partners that can reason through complex problems.

ThinkChain is my exploration of this idea, but it's really just the beginning. The best AI tools aren't just smart - they're tools that make AI smarter.

Fork it, extend it, and build something amazing. I can't wait to see what you create.