#!/usr/bin/env python3
"""
Test runner script for ThinkChain LLM adapter tests.

This script provides a convenient way to run different categories of tests
with appropriate configurations.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_api_keys():
    """Check which API keys are available."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    print("🔑 API Key Status:")
    print(f"  Anthropic: {'✅ Available' if anthropic_key else '❌ Missing'}")
    print(f"  Google:    {'✅ Available' if google_key else '❌ Missing'}")
    print()
    
    return anthropic_key, google_key


def run_pytest(args, description):
    """Run pytest with given arguments."""
    print(f"🧪 {description}")
    print(f"   Command: pytest {' '.join(args)}")
    print()
    
    try:
        result = subprocess.run(
            ["pytest"] + args,
            cwd=Path(__file__).parent.parent,  # Project root
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ Error: pytest not found. Install with: pip install pytest pytest-asyncio")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ThinkChain LLM adapter tests"
    )
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "all"],
        default="all",
        help="Test category to run (default: all)"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini", "all"],
        default="all",
        help="Provider to test (default: all)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--stop-on-failure",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    print("🚀 ThinkChain LLM Adapter Test Runner")
    print("=" * 40)
    
    # Check API keys
    anthropic_key, google_key = check_api_keys()
    
    # Build pytest arguments
    pytest_args = ["test/"]
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.stop_on_failure:
        pytest_args.append("-x")
    
    if args.coverage:
        pytest_args.extend(["--cov=llm_adapters", "--cov-report=term-missing"])
    
    # Filter by category
    if args.category == "unit":
        pytest_args = ["test/test_adapter_units.py"] + pytest_args[1:]
        description = "Running unit tests (no API calls required)"
    elif args.category == "integration":
        pytest_args = ["test/test_llm_adapters_integration.py"] + pytest_args[1:]
        description = "Running integration tests (requires API keys)"
        
        if not (anthropic_key or google_key):
            print("⚠️  Warning: No API keys found. Integration tests will be skipped.")
            print("   Set ANTHROPIC_API_KEY and/or GOOGLE_API_KEY to run these tests.")
            print()
    else:
        description = "Running all tests"
    
    # Filter by provider
    if args.provider != "all":
        pytest_args.extend(["-k", args.provider])
        description += f" (provider: {args.provider})"
    
    # Run the tests
    success = run_pytest(pytest_args, description)
    
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. See output above for details.")
        
        # Provide helpful suggestions
        print("\n💡 Troubleshooting tips:")
        print("   • Check that API keys are valid")
        print("   • Verify network connectivity")
        print("   • Run with -v for more detailed output")
        print("   • Run unit tests only: python test/run_tests.py --category unit")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 