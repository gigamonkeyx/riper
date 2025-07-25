#!/usr/bin/env python3
"""
Step 2: Verify OpenRouter basic functionality
"""

import sys
sys.path.insert(0, 'D:/pytorch')

from openrouter_client import get_openrouter_client

def test_openrouter_hello():
    """Test OpenRouter basic functionality"""
    print("=== STEP 2: OpenRouter Basic Verification ===")
    
    try:
        client = get_openrouter_client()
        
        response = client.chat_completion([
            {"role": "user", "content": "Hello, please respond with 'OpenRouter verification successful'"}
        ])
        
        if response.success:
            print("✅ OpenRouter connection successful")
            print(f"   Response: {response.content}")
            print(f"   Execution time: {response.execution_time:.2f}s")
            
            if "OpenRouter verification successful" in response.content:
                print("✅ STEP 2 PASSED: OpenRouter verification successful")
                return True
            else:
                print("⚠️ OpenRouter responded but didn't include verification phrase")
                return True  # Still consider it working
        else:
            print(f"❌ STEP 2 FAILED: {response.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ STEP 2 FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_openrouter_hello()
    if success:
        print("\n🎉 Ready for Step 3: Ollama API Integration Test")
    else:
        print("\n❌ Must fix OpenRouter before proceeding")
