#!/usr/bin/env python3
"""
Test Qwen2.5-coder with specific job description
"""

import ollama

def test_qwen25_with_job():
    """Test Qwen2.5-coder with specific RIPER-Ω job description"""
    
    # Specific job description for RIPER-Ω system
    system_prompt = """You are a RIPER-Ω System Task Breakdown Specialist.

YOUR EXACT JOB:
1. Take a RIPER-Ω system task description
2. Generate a numbered implementation checklist
3. Focus ONLY on evolutionary algorithms, bakery workflows, and agent coordination
4. NEVER interpret "evo" as EVE Online video game

RIPER-Ω CONTEXT:
- Rural bakery production simulation with 6 workflows
- Evolutionary algorithms optimize neural networks (fitness >70%)
- Agent-Based Modeling with Observer/Builder agents
- GPU-accelerated (RTX 3080) with A2A communication

OUTPUT: Numbered technical checklist with validation steps."""

    user_prompt = """RIPER-Ω SYSTEM TASK: Simple evo start test

YOUR JOB: Create a technical implementation checklist for this RIPER-Ω system task.

CONTEXT: This is about evolutionary algorithms and bakery production simulation, NOT video games.

Generate numbered steps for RIPER-Ω system implementation with validation checkpoints."""

    print('🔍 Testing Qwen2.5-coder with specific job description...')
    print('📋 Task: "Simple evo start test"')
    print('🎯 Expected: Evolutionary algorithm checklist, NOT EVE Online')
    print('-' * 60)
    
    try:
        response = ollama.chat(
            model='qwen2.5-coder:7b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        
        print('✅ Qwen2.5-coder Response:')
        print('=' * 60)
        print(response['message']['content'])
        print('=' * 60)
        
        # Check if response mentions EVE Online (bad) or evolutionary algorithms (good)
        content = response['message']['content'].lower()
        
        if 'eve online' in content or 'eve client' in content or 'character creation' in content:
            print('❌ FAILED: Still interpreting as EVE Online video game')
            return False
        elif 'evolutionary' in content or 'neural network' in content or 'fitness' in content:
            print('✅ SUCCESS: Correctly interpreted as evolutionary algorithms')
            return True
        else:
            print('⚠️ UNCLEAR: Response doesn\'t clearly indicate interpretation')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == '__main__':
    success = test_qwen25_with_job()
    print(f'\n🎯 Result: {"SUCCESS" if success else "NEEDS IMPROVEMENT"}')
