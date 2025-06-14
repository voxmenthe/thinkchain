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
#     "rich>=13.0.0",
#     "prompt-toolkit>=3.0.0",
# ]
# ///

"""
Smart launcher for ThinkChain

This script automatically detects available UI libraries and launches
the best available version of the application.
"""

import sys
import importlib

def check_dependencies():
    """Check which UI libraries are available"""
    deps = {
        'rich': False,
        'prompt_toolkit': False,
        'thinkchain': False,
        'thinkchain_cli': False
    }
    
    try:
        importlib.import_module('rich')
        deps['rich'] = True
    except ImportError:
        pass
    
    try:
        importlib.import_module('prompt_toolkit')
        deps['prompt_toolkit'] = True
    except ImportError:
        pass
    
    try:
        importlib.import_module('thinkchain')
        deps['thinkchain'] = True
    except ImportError:
        pass
    
    try:
        importlib.import_module('thinkchain_cli')
        deps['thinkchain_cli'] = True
    except ImportError:
        pass
    
    return deps

def main():
    """Launch the best available version"""
    deps = check_dependencies()
    
    print("🚀 Starting ThinkChain...")
    
    # Determine which version to launch
    if deps['thinkchain'] and deps['rich']:
        print("✨ Launching enhanced UI version with rich formatting")
        try:
            from thinkchain import interactive_chat
            interactive_chat()
        except Exception as e:
            print(f"⚠️  Enhanced UI failed ({e}), falling back to CLI version")
            if deps['thinkchain_cli']:
                from thinkchain_cli import interactive_chat
                interactive_chat()
            else:
                print("❌ No working version found")
                sys.exit(1)
    
    elif deps['thinkchain_cli']:
        print("🔧 Launching CLI version")
        from thinkchain_cli import interactive_chat
        interactive_chat()
    
    else:
        print("❌ No working version found. Please check your installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()