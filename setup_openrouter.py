"""
OpenRouter Setup and Configuration for RIPER-Ω System
Run this script to configure OpenRouter API integration
"""

import os
import sys
from openrouter_client import setup_openrouter_config, get_openrouter_client

def setup_openrouter():
    """Interactive setup for OpenRouter configuration"""
    print("=" * 60)
    print("RIPER-Ω OpenRouter Configuration Setup")
    print("=" * 60)
    
    print("\nThis script will configure OpenRouter API integration for Qwen3-Coder.")
    print("You'll need an OpenRouter API key from: https://openrouter.ai/")
    
    # Get API key
    api_key = input("\nEnter your OpenRouter API key: ").strip()
    
    if not api_key or api_key == "your-api-key-here":
        print("❌ Invalid API key. Please get a valid key from OpenRouter.")
        return False
    
    # Choose model
    print("\nAvailable Qwen3 models:")
    print("1. qwen/qwen-2.5-coder-32b-instruct (Recommended)")
    print("2. qwen/qwen-2.5-coder-14b-instruct")
    print("3. qwen/qwen-2.5-coder-7b-instruct")
    
    model_choice = input("Select model (1-3, default: 1): ").strip() or "1"
    
    models = {
        "1": "qwen/qwen-2.5-coder-32b-instruct",
        "2": "qwen/qwen-2.5-coder-14b-instruct", 
        "3": "qwen/qwen-2.5-coder-7b-instruct"
    }
    
    selected_model = models.get(model_choice, models["1"])
    
    # Setup configuration
    setup_openrouter_config(api_key, selected_model)
    
    print(f"\n✅ OpenRouter configured:")
    print(f"   Model: {selected_model}")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test connection
    print("\n🔄 Testing OpenRouter connection...")
    
    try:
        client = get_openrouter_client()
        success = client.test_connection()
        
        if success:
            print("✅ OpenRouter connection successful!")
            print("✅ RIPER-Ω is ready for hybrid OpenRouter + Ollama operation")
            return True
        else:
            print("❌ OpenRouter connection failed. Check your API key and internet connection.")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def create_env_file():
    """Create .env file for persistent configuration"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct")
    
    if api_key and api_key != "your-api-key-here":
        env_content = f"""# RIPER-Ω OpenRouter Configuration
OPENROUTER_API_KEY={api_key}
OPENROUTER_MODEL={model}

# Optional: Adjust these settings
OPENROUTER_MAX_TOKENS=4096
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_TIMEOUT=30
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Configuration saved to .env file")
        return True
    
    return False

def verify_integration():
    """Verify OpenRouter integration with RIPER-Ω components"""
    print("\n🔄 Verifying RIPER-Ω integration...")
    
    try:
        # Test FitnessScorer integration
        sys.path.insert(0, 'D:/pytorch')
        from agents import FitnessScorer
        
        scorer = FitnessScorer()
        print("✅ FitnessScorer with OpenRouter integration loaded")
        
        # Test orchestration integration
        from orchestration import Observer, Builder
        
        observer = Observer()
        builder = Builder()
        print("✅ Observer/Builder agents with OpenRouter integration loaded")
        
        # Test basic fitness evaluation
        test_result = scorer.process_task(
            {"test": "integration_check"}, 
            generation=0, 
            current_fitness=0.5
        )
        
        if test_result.success:
            print("✅ Hybrid fitness evaluation test successful")
            print(f"   Fitness score: {test_result.data.get('fitness_score', 'N/A')}")
        else:
            print("⚠️ Fitness evaluation test had issues (may be normal without API key)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration verification failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Starting RIPER-Ω OpenRouter integration setup...\n")
    
    # Step 1: Configure OpenRouter
    if not setup_openrouter():
        print("\n❌ OpenRouter setup failed. Please try again.")
        return
    
    # Step 2: Create .env file
    create_env_file()
    
    # Step 3: Verify integration
    verify_integration()
    
    print("\n" + "=" * 60)
    print("RIPER-Ω OpenRouter Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python test_openrouter_integration.py' to test the integration")
    print("2. Start your RIPER-Ω system with hybrid OpenRouter + Ollama support")
    print("3. Monitor fitness evaluations for improved performance")
    
    print("\nHybrid Architecture:")
    print("• OpenRouter (Qwen3): Cloud-based intelligent analysis")
    print("• Ollama: Local GPU-optimized processing")
    print("• Combined: Best of both worlds for >70% fitness achievement")

if __name__ == "__main__":
    main()
