#!/usr/bin/env python3
"""
Step 3: Verify Ollama API integration (the exact method used by hybrid system)
"""

import sys
import requests
import json
sys.path.insert(0, 'D:/pytorch')

from agents import FitnessScorer

def test_ollama_api_direct():
    """Test Ollama API directly using the same method as hybrid system"""
    print("=== STEP 3A: Direct Ollama API Test ===")
    
    try:
        payload = {
            "model": "qwen3:8b",
            "prompt": "Hello, please respond with 'Ollama API verification successful'",
            "stream": False,
            "options": {
                "num_gpu": 1,
                "num_thread": 8,
                "temperature": 0.7
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            print("✅ Direct Ollama API call successful")
            print(f"   Response: {response_text[:100]}...")
            
            if "Ollama API verification successful" in response_text:
                print("✅ STEP 3A PASSED: Direct API verification successful")
                return True
            else:
                print("⚠️ API responded but didn't include verification phrase")
                return True  # Still working
        else:
            print(f"❌ STEP 3A FAILED: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ STEP 3A FAILED: {e}")
        return False

def test_ollama_via_fitness_scorer():
    """Test Ollama via FitnessScorer _call_ollama method"""
    print("\n=== STEP 3B: FitnessScorer Ollama Integration Test ===")
    
    try:
        scorer = FitnessScorer()
        print(f"✅ FitnessScorer initialized with model: {scorer.model_name}")
        print(f"   Ollama available: {scorer.ollama_available}")
        print(f"   GPU enabled: {scorer.gpu_enabled}")
        
        # Test the exact _call_ollama method used by hybrid system
        result = scorer._call_ollama(
            "Hello, please respond with 'FitnessScorer Ollama verification successful'",
            "You are a helpful AI assistant."
        )
        
        if "error" not in result:
            response_text = result.get("response", "")
            print("✅ FitnessScorer Ollama call successful")
            print(f"   Response: {response_text[:100]}...")
            print(f"   Execution time: {result.get('execution_time', 'N/A'):.2f}s")
            print(f"   Tokens/sec: {result.get('tokens_per_second', 'N/A'):.1f}")
            
            if "FitnessScorer Ollama verification successful" in response_text:
                print("✅ STEP 3B PASSED: FitnessScorer integration successful")
                return True
            else:
                print("⚠️ FitnessScorer responded but didn't include verification phrase")
                return True  # Still working
        else:
            print(f"❌ STEP 3B FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ STEP 3B FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama API integration step-by-step...\n")
    
    step3a = test_ollama_api_direct()
    step3b = test_ollama_via_fitness_scorer()
    
    if step3a and step3b:
        print("\n🎉 STEP 3 PASSED: Ready for Step 4 (Full Hybrid Test)")
    else:
        print("\n❌ STEP 3 FAILED: Must fix Ollama integration before hybrid test")
