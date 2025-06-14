"""
MCP (Model Context Protocol) Integration for ThinkChain

This module extends the existing tool discovery system to include MCP servers
while preserving all existing functionality including streaming and thinking.
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP packages not available. MCP functionality will be disabled.")

# Setup logging
logger = logging.getLogger(__name__)


class MCPTool:
    """Represents a tool from an MCP server, compatible with our tool registry"""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any], server_name: str):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.server_name = server_name
    
    def to_claude_format(self) -> Dict[str, Any]:
        """Convert to Claude API tool format"""
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema
        }


class MCPServer:
    """Manages individual MCP server connections"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()
        self.tools: List[MCPTool] = []
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the MCP server connection"""
        if not MCP_AVAILABLE:
            logger.warning(f"MCP not available, skipping server {self.name}")
            return False
            
        if not self.config.get('enabled', False):
            logger.info(f"MCP server {self.name} is disabled")
            return False
        
        try:
            command = self.config.get('command')
            if command == 'npx':
                command = shutil.which('npx')
            elif command == 'uvx':
                command = shutil.which('uvx')
            else:
                command = shutil.which(command)
            
            if not command:
                logger.error(f"Command not found for MCP server {self.name}")
                return False
            
            server_params = StdioServerParameters(
                command=command,
                args=self.config.get('args', []),
                env={**os.environ, **self.config.get('env', {})}
            )
            
            # Set up stdio transport
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            
            # Create session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize session
            await session.initialize()
            self.session = session
            
            # Discover tools
            await self._discover_tools()
            
            self.is_initialized = True
            logger.info(f"Initialized MCP server {self.name} with {len(self.tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server {self.name}: {e}")
            await self.cleanup()
            return False
    
    async def _discover_tools(self):
        """Discover tools from the MCP server"""
        if not self.session:
            return
        
        try:
            tools_response = await self.session.list_tools()
            self.tools.clear()
            
            for item in tools_response:
                if isinstance(item, tuple) and item[0] == "tools":
                    for tool in item[1]:
                        mcp_tool = MCPTool(
                            name=f"mcp_{self.name}_{tool.name}",  # Prefix to avoid conflicts
                            description=f"[MCP:{self.name}] {tool.description}",
                            input_schema=tool.inputSchema,
                            server_name=self.name
                        )
                        self.tools.append(mcp_tool)
            
        except Exception as e:
            logger.error(f"Failed to discover tools for {self.name}: {e}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool on this MCP server"""
        if not self.session or not self.is_initialized:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        
        # Remove the mcp prefix to get the original tool name
        original_tool_name = tool_name.replace(f"mcp_{self.name}_", "")
        
        try:
            logger.info(f"Executing MCP tool {original_tool_name} on server {self.name}")
            result = await self.session.call_tool(original_tool_name, arguments)
            
            # Convert result to string format expected by our system
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            raise
    
    async def cleanup(self):
        """Clean up server resources"""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.is_initialized = False
                logger.info(f"Cleaned up MCP server {self.name}")
            except Exception as e:
                logger.error(f"Error during cleanup of MCP server {self.name}: {e}")


class MCPManager:
    """Manages all MCP servers and integrates with our tool discovery system"""
    
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        self.servers: Dict[str, MCPServer] = {}
        self.mcp_tools: List[MCPTool] = []
        self.is_initialized = False
    
    def load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(f"MCP config file {self.config_path} not found")
            return {"mcpServers": {}}
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading MCP config: {e}")
            return {"mcpServers": {}}
    
    async def initialize(self) -> int:
        """Initialize all MCP servers"""
        if not MCP_AVAILABLE:
            logger.info("MCP not available, skipping MCP initialization")
            return 0
        
        config = self.load_config()
        mcp_servers_config = config.get("mcpServers", {})
        
        if not mcp_servers_config:
            logger.info("No MCP servers configured")
            return 0
        
        # Create server instances
        self.servers = {
            name: MCPServer(name, server_config)
            for name, server_config in mcp_servers_config.items()
        }
        
        # Initialize servers concurrently
        initialization_tasks = [
            server.initialize() for server in self.servers.values()
        ]
        
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Collect tools from successfully initialized servers
        self.mcp_tools.clear()
        initialized_count = 0
        
        for server, result in zip(self.servers.values(), results):
            if isinstance(result, bool) and result:
                self.mcp_tools.extend(server.tools)
                initialized_count += 1
            elif isinstance(result, Exception):
                logger.error(f"Server {server.name} failed to initialize: {result}")
        
        self.is_initialized = True
        logger.info(f"Initialized {initialized_count} MCP servers with {len(self.mcp_tools)} tools total")
        return len(self.mcp_tools)
    
    def get_claude_tools(self) -> List[Dict[str, Any]]:
        """Get MCP tools in Claude API format"""
        return [tool.to_claude_format() for tool in self.mcp_tools]
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute an MCP tool"""
        # Find which server this tool belongs to
        for server in self.servers.values():
            for tool in server.tools:
                if tool.name == tool_name:
                    return await server.execute_tool(tool_name, tool_input)
        
        raise ValueError(f"MCP tool {tool_name} not found")
    
    def list_tools(self) -> List[str]:
        """List all MCP tool names"""
        return [tool.name for tool in self.mcp_tools]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific MCP tool"""
        for tool in self.mcp_tools:
            if tool.name == tool_name:
                return {
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.input_schema,
                    'server': tool.server_name,
                    'type': 'mcp'
                }
        return None
    
    async def cleanup(self):
        """Clean up all MCP servers"""
        if self.servers:
            cleanup_tasks = [server.cleanup() for server in self.servers.values()]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.servers.clear()
        self.mcp_tools.clear()
        self.is_initialized = False
        logger.info("MCP manager cleanup completed")
    
    async def refresh(self) -> int:
        """Refresh MCP servers (useful for development)"""
        await self.cleanup()
        return await self.initialize()


# Global MCP manager instance
_mcp_manager: Optional[MCPManager] = None

def get_mcp_manager() -> MCPManager:
    """Get the global MCP manager instance"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager

async def initialize_mcp() -> int:
    """Initialize MCP integration"""
    return await get_mcp_manager().initialize()

async def cleanup_mcp():
    """Clean up MCP integration"""
    if _mcp_manager:
        await _mcp_manager.cleanup()

def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get all MCP tools in Claude API format"""
    return get_mcp_manager().get_claude_tools()

async def execute_mcp_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute an MCP tool"""
    return await get_mcp_manager().execute_tool(tool_name, tool_input)

def list_mcp_tools() -> List[str]:
    """List all MCP tool names"""
    return get_mcp_manager().list_tools()

async def refresh_mcp() -> int:
    """Refresh MCP servers"""
    return await get_mcp_manager().refresh()