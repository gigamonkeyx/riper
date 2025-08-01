#!/usr/bin/env python3
"""
Test script to verify Ollama handoff fix
Tests local Ollama Qwen3 instead of OpenRouter
"""

import logging
import sys
import ollama
from orchestration import Observer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test local Ollama connection"""
    print("🔍 Testing local Ollama connection...")
    
    try:
        # Test basic Ollama connection
        models = ollama.list()
        print(f"✅ Ollama connected. Available models: {len(models['models'])}")
        
        # Check if qwen3:8b is available
        qwen3_available = any(model['name'] == 'qwen3:8b' for model in models['models'])
        if qwen3_available:
            print("✅ Qwen3:8b model is available locally")
        else:
            print("❌ Qwen3:8b model not found. Available models:")
            for model in models['models']:
                print(f"   - {model['name']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_qwen3_chat():
    """Test Qwen3 chat functionality"""
    print("\n🔍 Testing Qwen3 chat functionality...")
    
    try:
        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please respond with 'Qwen3 local test successful'"}
            ]
        )
        
        if response and 'message' in response:
            content = response['message']['content']
            print(f"✅ Qwen3 response: {content}")
            
            if "successful" in content.lower():
                print("✅ Qwen3 chat test passed")
                return True
            else:
                print("⚠️ Qwen3 responded but didn't include expected text")
                return False
        else:
            print("❌ No valid response from Qwen3")
            return False
            
    except Exception as e:
        print(f"❌ Qwen3 chat test failed: {e}")
        return False

def test_observer_handoff():
    """Test Observer handoff functionality"""
    print("\n🔍 Testing Observer handoff functionality...")
    
    try:
        # Create Observer instance
        observer = Observer("test_observer")
        print("✅ Observer instance created")
        
        # Test handoff method
        result = observer.openrouter_to_ollama_handoff(
            "Test task: Generate a simple Python function that adds two numbers",
            "qwen3:8b"
        )
        
        if result and result.get('success', False):
            print("✅ Observer handoff test passed")
            print(f"   Handoff result keys: {list(result.keys())}")
            return True
        else:
            print(f"❌ Observer handoff failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Observer handoff test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 OLLAMA HANDOFF FIX VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Qwen3 Chat", test_qwen3_chat),
        ("Observer Handoff", test_observer_handoff)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n🎯 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ollama handoff fix is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
