#!/usr/bin/env python3
"""
Setup API Keys for Reflective Coherence Explorer

This script guides users through setting up API keys for OpenAI and/or Claude
integration to enable AI-powered insights and explanations.
"""

import os
import sys
from getpass import getpass
import subprocess
from pathlib import Path

def print_header():
    """Print script header."""
    print("\n" + "=" * 60)
    print("🔑 Reflective Coherence Explorer - API Key Setup")
    print("=" * 60)

def print_instructions():
    """Print instructions for getting API keys."""
    print("\nThis script will help you set up API keys for AI-powered features.")
    print("\nYou'll need at least one of the following:")
    print("  1. OpenAI API key (https://platform.openai.com/account/api-keys)")
    print("  2. Claude API key (https://console.anthropic.com/keys)")
    print("\nYou can press Enter to skip either key if you don't have it.")
    print("Your keys will be stored securely in a local .env file.\n")

def get_api_keys():
    """Get API keys from user."""
    openai_key = getpass("Enter your OpenAI API key (or press Enter to skip): ")
    claude_key = getpass("Enter your Claude API key (or press Enter to skip): ")
    
    return openai_key, claude_key

def save_api_keys(openai_key, claude_key):
    """Save API keys to .env file."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env.local"
    
    with open(env_file, "w") as f:
        if openai_key:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
        if claude_key:
            f.write(f"CLAUDE_API_KEY={claude_key}\n")
    
    # Set permissions to restrict access to only the user
    os.chmod(env_file, 0o600)
    
    return env_file

def check_existing_keys():
    """Check if API keys are already set up."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env.local"
    
    if env_file.exists():
        with open(env_file, "r") as f:
            content = f.read()
            openai_set = "OPENAI_API_KEY" in content
            claude_set = "CLAUDE_API_KEY" in content
            
        if openai_set or claude_set:
            print("\n⚠️  API keys already set up in .env.local")
            print("   - OpenAI key:", "✓ Set" if openai_set else "✗ Not set")
            print("   - Claude key:", "✓ Set" if claude_set else "✗ Not set")
            
            overwrite = input("\nDo you want to overwrite the existing keys? (y/n): ").lower().strip()
            return overwrite != "y"
    
    return False

def setup_environment():
    """Set up environment variables for current session."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env.local"
    
    if not env_file.exists():
        return
    
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value
    
    print("\n✅ Environment variables set for current session.")

def load_env_dot_file(env_file):
    """Read key-value pairs from .env file."""
    env_vars = {}
    
    if not env_file.exists():
        return env_vars
    
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
                except ValueError:
                    pass  # Skip malformed lines
    
    return env_vars

def show_setup_instructions():
    """Show instructions for using the API keys."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env.local"
    
    if not env_file.exists():
        return
    
    env_vars = load_env_dot_file(env_file)
    openai_set = "OPENAI_API_KEY" in env_vars
    claude_set = "CLAUDE_API_KEY" in env_vars
    
    print("\n📋 API Key Setup Complete")
    print("   - OpenAI key:", "✓ Set" if openai_set else "✗ Not set")
    print("   - Claude key:", "✓ Set" if claude_set else "✗ Not set")
    
    print("\nTo use these keys with the Reflective Coherence Explorer:")
    
    # Instructions for different shells
    print("\nFor Bash/Zsh:")
    if openai_set:
        print(f"  export OPENAI_API_KEY='{env_vars['OPENAI_API_KEY']}'")
    if claude_set:
        print(f"  export CLAUDE_API_KEY='{env_vars['CLAUDE_API_KEY']}'")
    
    print("\nFor Windows Command Prompt:")
    if openai_set:
        print(f"  set OPENAI_API_KEY={env_vars['OPENAI_API_KEY']}")
    if claude_set:
        print(f"  set CLAUDE_API_KEY={env_vars['CLAUDE_API_KEY']}")
    
    print("\nFor PowerShell:")
    if openai_set:
        print(f"  $env:OPENAI_API_KEY = '{env_vars['OPENAI_API_KEY']}'")
    if claude_set:
        print(f"  $env:CLAUDE_API_KEY = '{env_vars['CLAUDE_API_KEY']}'")
    
    print("\nOr, use the run.py script which will load these keys automatically.")

def main():
    """Main function."""
    print_header()
    print_instructions()
    
    # Check for existing keys
    if check_existing_keys():
        print("\n⚠️  Setup cancelled. Existing keys will be used.")
        return
    
    # Get API keys
    openai_key, claude_key = get_api_keys()
    
    # Validate that at least one key is provided
    if not openai_key and not claude_key:
        print("\n❌ No API keys provided. Setup cancelled.")
        print("   The application will run with limited functionality.")
        return
    
    # Save API keys
    env_file = save_api_keys(openai_key, claude_key)
    print(f"\n✅ API keys saved to {env_file}")
    
    # Setup environment for current session
    setup_environment()
    
    # Show setup instructions
    show_setup_instructions()
    
    print("\n🚀 You're all set to use AI-powered features in the Reflective Coherence Explorer!")

if __name__ == "__main__":
    main() 