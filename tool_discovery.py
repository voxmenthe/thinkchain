"""
Tool Discovery System for ThinkChain

This module automatically discovers and loads tool classes from the tools directory,
validates them against the BaseTool interface, and creates the tool registry for
use with Claude's API.
"""

import importlib
import inspect
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Type, Optional
import logging

from tools.base import BaseTool

# Import MCP integration (optional)
try:
    from mcp_integration import (
        get_mcp_manager, initialize_mcp, get_mcp_tools, 
        execute_mcp_tool, list_mcp_tools, refresh_mcp
    )
    MCP_INTEGRATION_AVAILABLE = True
except ImportError:
    MCP_INTEGRATION_AVAILABLE = False
    logging.info("MCP integration not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolDiscovery:
    """Discovers and manages tools from the tools directory and MCP servers"""
    
    def __init__(self, tools_dir: str = "tools"):
        self.tools_dir = Path(tools_dir)
        self.discovered_tools: Dict[str, Type[BaseTool]] = {}
        self.tool_registry: Dict[str, Any] = {}
        self.mcp_tools: List[Dict[str, Any]] = []
        self.mcp_initialized = False
        
    def discover_tools(self) -> Dict[str, Type[BaseTool]]:
        """
        Discover all tool classes in the tools directory that inherit from BaseTool
        
        Returns:
            Dict mapping tool names to tool classes
        """
        self.discovered_tools.clear()
        
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory {self.tools_dir} does not exist")
            return {}
            
        # Add tools directory to Python path if not already there
        tools_path = str(self.tools_dir.parent.absolute())
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
            
        # Iterate through all Python files in tools directory
        for tool_file in self.tools_dir.glob("*.py"):
            if tool_file.name.startswith("_") or tool_file.name == "base.py":
                continue
                
            module_name = f"tools.{tool_file.stem}"
            
            try:
                # Import the module
                if module_name in sys.modules:
                    # Reload if already imported to pick up changes
                    module = importlib.reload(sys.modules[module_name])
                else:
                    module = importlib.import_module(module_name)
                
                # Find classes that inherit from BaseTool
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (obj != BaseTool and 
                        issubclass(obj, BaseTool) and 
                        obj.__module__ == module_name):
                        
                        # Validate the tool
                        if self._validate_tool(obj):
                            tool_instance = obj()
                            self.discovered_tools[tool_instance.name] = obj
                            logger.info(f"Discovered tool: {tool_instance.name}")
                        else:
                            logger.warning(f"Tool {name} failed validation")
                            
            except Exception as e:
                logger.error(f"Error importing {module_name}: {e}")
                
        logger.info(f"Discovered {len(self.discovered_tools)} tools")
        return self.discovered_tools
    
    def _validate_tool(self, tool_class: Type[BaseTool]) -> bool:
        """
        Validate that a tool class properly implements the BaseTool interface
        
        Args:
            tool_class: The tool class to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to instantiate the tool
            tool_instance = tool_class()
            
            # Check required properties exist and are not None
            if not hasattr(tool_instance, 'name') or not tool_instance.name:
                logger.error(f"Tool {tool_class.__name__} missing or empty 'name' property")
                return False
                
            if not hasattr(tool_instance, 'description') or not tool_instance.description:
                logger.error(f"Tool {tool_class.__name__} missing or empty 'description' property")
                return False
                
            if not hasattr(tool_instance, 'input_schema') or not tool_instance.input_schema:
                logger.error(f"Tool {tool_class.__name__} missing or empty 'input_schema' property")
                return False
                
            # Check that execute method exists and is callable
            if not hasattr(tool_instance, 'execute') or not callable(tool_instance.execute):
                logger.error(f"Tool {tool_class.__name__} missing or non-callable 'execute' method")
                return False
                
            # Validate tool name format (alphanumeric, underscore, dash only)
            import re
            if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', tool_instance.name):
                logger.error(f"Tool {tool_class.__name__} has invalid name format: {tool_instance.name}")
                return False
                
            # Validate input_schema is a dict with required structure
            schema = tool_instance.input_schema
            if not isinstance(schema, dict):
                logger.error(f"Tool {tool_class.__name__} input_schema must be a dictionary")
                return False
                
            if schema.get('type') != 'object':
                logger.error(f"Tool {tool_class.__name__} input_schema must have type 'object'")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating tool {tool_class.__name__}: {e}")
            return False
    
    def create_tool_registry(self) -> Dict[str, Any]:
        """
        Create the tool registry in the format expected by Claude's API
        
        Returns:
            Dictionary mapping tool names to their schema definitions
        """
        self.tool_registry.clear()
        
        for tool_name, tool_class in self.discovered_tools.items():
            try:
                tool_instance = tool_class()
                self.tool_registry[tool_name] = {
                    'name': tool_instance.name,
                    'description': tool_instance.description,
                    'input_schema': tool_instance.input_schema,
                    'class': tool_class
                }
            except Exception as e:
                logger.error(f"Error creating registry entry for {tool_name}: {e}")
                
        return self.tool_registry
    
    async def initialize_mcp(self) -> int:
        """Initialize MCP integration"""
        if not MCP_INTEGRATION_AVAILABLE:
            logger.info("MCP integration not available")
            return 0
        
        try:
            mcp_tool_count = await initialize_mcp()
            self.mcp_tools = get_mcp_tools()
            self.mcp_initialized = True
            logger.info(f"Initialized MCP integration with {mcp_tool_count} tools")
            return mcp_tool_count
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            return 0
    
    def get_claude_tools(self) -> List[Dict[str, Any]]:
        """
        Get tools in the format expected by Claude's Messages API
        
        Returns:
            List of tool definitions for Claude API (local + MCP tools)
        """
        claude_tools = []
        
        # Add local tools
        for tool_name, tool_info in self.tool_registry.items():
            claude_tool = {
                'name': tool_info['name'],
                'description': tool_info['description'],
                'input_schema': tool_info['input_schema']
            }
            claude_tools.append(claude_tool)
        
        # Add MCP tools
        claude_tools.extend(self.mcp_tools)
            
        return claude_tools
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a discovered tool with the given input (local or MCP)
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            Tool execution result as string
        """
        # Try local tools first
        if tool_name in self.tool_registry:
            try:
                tool_class = self.tool_registry[tool_name]['class']
                tool_instance = tool_class()
                result = tool_instance.execute(**tool_input)
                
                # Ensure result is a string
                if not isinstance(result, str):
                    result = str(result)
                    
                return result
                
            except Exception as e:
                logger.error(f"Error executing local tool {tool_name}: {e}")
                return f"Error executing local tool {tool_name}: {str(e)}"
        
        # Try MCP tools
        if self.mcp_initialized and MCP_INTEGRATION_AVAILABLE:
            try:
                mcp_tool_names = list_mcp_tools()
                if tool_name in mcp_tool_names:
                    result = await execute_mcp_tool(tool_name, tool_input)
                    return result
            except Exception as e:
                logger.error(f"Error executing MCP tool {tool_name}: {e}")
                return f"Error executing MCP tool {tool_name}: {str(e)}"
        
        return f"Error: Tool '{tool_name}' not found in local or MCP registries"
    
    async def refresh_tools(self) -> int:
        """
        Refresh the tool discovery (useful for development)
        
        Returns:
            Total number of tools discovered (local + MCP)
        """
        # Refresh local tools
        self.discover_tools()
        self.create_tool_registry()
        
        # Refresh MCP tools
        mcp_count = 0
        if MCP_INTEGRATION_AVAILABLE:
            try:
                mcp_count = await refresh_mcp()
                self.mcp_tools = get_mcp_tools()
            except Exception as e:
                logger.error(f"Error refreshing MCP tools: {e}")
        
        total_tools = len(self.discovered_tools) + len(self.mcp_tools)
        return total_tools
    
    def list_tools(self) -> List[str]:
        """
        Get a list of all discovered tool names (local + MCP)
        
        Returns:
            List of tool names
        """
        local_tools = list(self.discovered_tools.keys())
        mcp_tool_names = []
        
        if MCP_INTEGRATION_AVAILABLE and self.mcp_initialized:
            try:
                mcp_tool_names = list_mcp_tools()
            except Exception as e:
                logger.error(f"Error listing MCP tools: {e}")
        
        return local_tools + mcp_tool_names
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary
        """
        if tool_name not in self.tool_registry:
            return {}
            
        return self.tool_registry[tool_name]


# Global instance for easy access
_tool_discovery = None

def get_tool_discovery() -> ToolDiscovery:
    """Get the global ToolDiscovery instance"""
    global _tool_discovery
    if _tool_discovery is None:
        _tool_discovery = ToolDiscovery()
        _tool_discovery.discover_tools()
        _tool_discovery.create_tool_registry()
    return _tool_discovery

async def initialize_tool_discovery() -> int:
    """Initialize tool discovery including MCP integration"""
    discovery = get_tool_discovery()
    
    # Initialize MCP integration
    mcp_count = await discovery.initialize_mcp()
    total_local = len(discovery.discovered_tools)
    
    logger.info(f"Tool discovery initialized: {total_local} local tools, {mcp_count} MCP tools")
    return total_local + mcp_count

async def refresh_tools() -> int:
    """Refresh the global tool discovery"""
    return await get_tool_discovery().refresh_tools()

def get_claude_tools() -> List[Dict[str, Any]]:
    """Get all tools in Claude API format"""
    return get_tool_discovery().get_claude_tools()

async def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool by name (async to support MCP tools)"""
    return await get_tool_discovery().execute_tool(tool_name, tool_input)

def list_tools() -> List[str]:
    """List all available tool names"""
    return get_tool_discovery().list_tools()

# Synchronous wrapper for backward compatibility
def execute_tool_sync(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool by name (synchronous wrapper for legacy support)"""
    discovery = get_tool_discovery()
    
    # Try local tools first (these are synchronous)
    if tool_name in discovery.tool_registry:
        try:
            tool_class = discovery.tool_registry[tool_name]['class']
            tool_instance = tool_class()
            result = tool_instance.execute(**tool_input)
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing local tool {tool_name}: {e}")
            return f"Error executing local tool {tool_name}: {str(e)}"
    
    # For MCP tools, we need to run in an event loop
    if discovery.mcp_initialized and MCP_INTEGRATION_AVAILABLE:
        try:
            mcp_tool_names = list_mcp_tools()
            if tool_name in mcp_tool_names:
                # Run the async function in the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, execute_mcp_tool(tool_name, tool_input))
                        return future.result()
                else:
                    # No event loop running, we can use asyncio.run
                    return asyncio.run(execute_mcp_tool(tool_name, tool_input))
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return f"Error executing MCP tool {tool_name}: {str(e)}"
    
    return f"Error: Tool '{tool_name}' not found in local or MCP registries"