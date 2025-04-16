#!/usr/bin/env python3
"""
Reflective Coherence Explorer - Launch Script

This script provides a simple way to launch the Reflective Coherence Explorer 
application. It handles setting up the environment, checking dependencies,
and starting the Streamlit server.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import time
import importlib.util

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "streamlit", "numpy", "pandas", "matplotlib", 
        "scipy", "openai", "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def load_env_file():
    """Load environment variables from .env.local file."""
    env_file = Path(__file__).parent / ".env.local"
    
    if not env_file.exists():
        print("\n⚠️  No .env.local file found for API keys")
        print("   You can set up API keys by running: python setup_api_keys.py")
        print("   The application will still work, but without AI-powered insights.\n")
        return False
    
    # Load variables from .env file
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
                except ValueError:
                    continue  # Skip malformed lines
    
    # Check if any API keys were loaded
    openai_key = os.environ.get("OPENAI_API_KEY")
    claude_key = os.environ.get("CLAUDE_API_KEY")
    
    if openai_key or claude_key:
        print("\n✅ API keys loaded from .env.local file")
        print("   - OpenAI API key:", "✓ Loaded" if openai_key else "✗ Not found")
        print("   - Claude API key:", "✓ Loaded" if claude_key else "✗ Not found")
        return True
    else:
        print("\n⚠️  No API keys found in .env.local file")
        print("   You can set up API keys by running: python setup_api_keys.py")
        print("   The application will still work, but without AI-powered insights.\n")
        return False

def check_api_keys():
    """Check if API keys are set and provide warnings if not."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    claude_key = os.environ.get("CLAUDE_API_KEY")
    
    if not openai_key and not claude_key:
        print("\n⚠️  No API keys found for LLM integration")
        print("   For AI-enhanced features, either:")
        print("   1. Run the setup script: python setup_api_keys.py")
        print("   2. Set these environment variables manually:")
        print("      - OPENAI_API_KEY")
        print("      - CLAUDE_API_KEY")
        print("   The application will still work, but without AI-powered insights.\n")
    return

def launch_app():
    """Launch the Streamlit application."""
    # Get the path to the app
    project_root = Path(__file__).parent
    app_path = project_root / "app" / "dashboard" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find the application at {app_path}")
        return False
    
    # Start the Streamlit server
    print("\n🚀 Launching Reflective Coherence Explorer...")
    print(f"Starting Streamlit server for {app_path}")
    
    try:
        # Run streamlit as a subprocess
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(app_path)], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for the server to start
        for line in process.stdout:
            print(line, end="")
            if "You can now view your Streamlit app in your browser" in line:
                # Extract the URL
                for next_line in process.stdout:
                    if "http://" in next_line:
                        url = next_line.strip()
                        print(f"\n✨ Opening {url} in your browser...")
                        webbrowser.open(url)
                        break
                break
        
        # Keep the process running
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Reflective Coherence Explorer...")
        process.terminate()
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 Reflective Coherence (ΨC) Explorer")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Try to load API keys from .env.local
    load_env_file()
    
    # Final check of API keys
    check_api_keys()
    
    if not launch_app():
        sys.exit(1) 