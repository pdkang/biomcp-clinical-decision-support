#!/usr/bin/env python3
"""
Simple startup script for BioMCP API server using system Python
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Check if required environment variables are set
required_vars = ["ANTHROPIC_API_KEY", "BIOMCP_OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ Missing required environment variables: {missing_vars}")
    print("Please check your .env file and ensure all API keys are set.")
    sys.exit(1)

print("✅ Environment variables loaded successfully")
print(f"ANTHROPIC_API_KEY: {'*' * 10}{os.getenv('ANTHROPIC_API_KEY')[-4:] if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
print(f"BIOMCP_OPENAI_API_KEY: {'*' * 10}{os.getenv('BIOMCP_OPENAI_API_KEY')[-4:] if os.getenv('BIOMCP_OPENAI_API_KEY') else 'NOT SET'}")

# Import and start the server
try:
    from biomcp.api_server import app
    import uvicorn
    
    print("🚀 Starting BioMCP API server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    print("🧠 Autonomous thinking: http://localhost:8000/thinking/autonomous")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Server startup error: {e}")
    sys.exit(1)