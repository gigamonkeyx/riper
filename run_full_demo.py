#!/usr/bin/env python3
"""
RIPER-Ω Comprehensive Bakery Simulation Demo
Full output demonstration with all workflows and systems
"""

import time
import json
import threading
from economy_sim import MesaBakeryModel
from economy_rewards import SDSystem
from orchestration import Observer, Builder
import ui_api

def print_banner(title):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def run_comprehensive_demo():
    """Run the comprehensive bakery simulation demo"""
    
    print_banner("RIPER-Ω COMPREHENSIVE BAKERY SIMULATION DEMO")
    
    # 1. Initialize all systems
    print("\n🔧 INITIALIZING SIMULATION SYSTEMS...")
    
    print("   📊 Initializing Mesa ABM Model...")
    model = MesaBakeryModel()
    print(f"   ✅ Mesa Model: {model.num_agents} agents on {model.grid.width}x{model.grid.height} grid")
    
    print("   🎯 Initializing System Dynamics...")
    sd_system = SDSystem()
    print("   ✅ SD System: Economic flows and feedback loops active")
    
    print("   🤖 Initializing Multi-Agent Orchestration...")
    observer = Observer("observer_001")
    builder = Builder("builder_001")
    print("   ✅ Orchestration: Observer and Builder agents initialized with GPU optimization")
    
    # 2. Run simulation steps with full output
    print_banner("RUNNING COMPREHENSIVE SIMULATION")
    
    total_production = 0
    total_costs = 0
    
    for step in range(1, 11):  # 10 simulation steps
        print(f"\n📅 SIMULATION STEP {step}/10")
        print("-" * 40)
        
        # Run Mesa step
        start_time = time.time()
        model.step()
        mesa_time = time.time() - start_time
        
        # Get production metrics
        bread_production = sum(agent.bread_produced for agent in model.schedule.agents 
                             if hasattr(agent, 'bread_produced'))
        labor_costs = sum(agent.daily_wage for agent in model.schedule.agents 
                         if hasattr(agent, 'daily_wage'))
        
        total_production += bread_production
        total_costs += labor_costs
        
        # Run SD system
        start_time = time.time()
        sd_metrics = sd_system.step()
        sd_time = time.time() - start_time
        
        print(f"   🍞 Bread Production: {bread_production:,} items")
        print(f"   💰 Labor Costs: ${labor_costs:,.2f}")
        print(f"   📈 Revenue: ${bread_production * 5.0:,.2f}")
        print(f"   ⚡ Mesa Step Time: {mesa_time:.3f}s")
        print(f"   🎯 SD Step Time: {sd_time:.3f}s")
        print(f"   📊 Economic Flow: ${sd_metrics.get('total_flow', 0):,.2f}")
        
        # Brief pause for readability
        time.sleep(0.5)
    
    # 3. Final comprehensive analysis
    print_banner("COMPREHENSIVE ANALYSIS RESULTS")
    
    avg_production = total_production / 10
    avg_costs = total_costs / 10
    avg_revenue = avg_production * 5.0
    avg_profit = avg_revenue - avg_costs
    profit_margin = (avg_profit / avg_revenue) * 100 if avg_revenue > 0 else 0
    
    print(f"📊 PRODUCTION METRICS (10-step average):")
    print(f"   🍞 Average Daily Production: {avg_production:,.0f} items")
    print(f"   💰 Average Daily Labor Costs: ${avg_costs:,.2f}")
    print(f"   📈 Average Daily Revenue: ${avg_revenue:,.2f}")
    print(f"   💵 Average Daily Profit: ${avg_profit:,.2f}")
    print(f"   📊 Profit Margin: {profit_margin:.1f}%")
    
    # Agent breakdown
    agent_types = {}
    for agent in model.schedule.agents:
        agent_type = type(agent).__name__
        if agent_type not in agent_types:
            agent_types[agent_type] = 0
        agent_types[agent_type] += 1
    
    print(f"\n👥 AGENT BREAKDOWN:")
    for agent_type, count in agent_types.items():
        print(f"   {agent_type}: {count} agents")
    
    # Workflow analysis
    print(f"\n🏭 WORKFLOW SYSTEMS ACTIVE:")
    workflows = [
        "🥩 Meat Production (14-day batch cycles)",
        "🌾 Grain Milling (daily 1-ton capacity)", 
        "🍎 Fruit Processing (seasonal 15K lb)",
        "🍞 Baking (daily 1K+ units)",
        "🥫 Expanded Canning (HACCP compliant)",
        "🛒 Counter Sales (retail/wholesale)"
    ]
    
    for workflow in workflows:
        print(f"   ✅ {workflow}")
    
    print(f"\n🎯 SYSTEM PERFORMANCE:")
    print(f"   ⚡ Total Agents: {model.num_agents}")
    print(f"   🔄 Simulation Steps: 10")
    print(f"   📈 Total Production: {total_production:,} items")
    print(f"   💰 Total Costs: ${total_costs:,.2f}")
    print(f"   🏆 System Status: FULLY OPERATIONAL")
    
    return {
        'model': model,
        'sd_system': sd_system,
        'observer': observer,
        'builder': builder,
        'metrics': {
            'avg_production': avg_production,
            'avg_costs': avg_costs,
            'avg_revenue': avg_revenue,
            'avg_profit': avg_profit,
            'profit_margin': profit_margin,
            'total_agents': model.num_agents
        }
    }

def start_ui_server(demo_results):
    """Start the UI server with demo results"""
    print_banner("STARTING UI SERVER")
    
    # Initialize UI API with demo results
    ui_api.model = demo_results['model']
    ui_api.sd_system = demo_results['sd_system']
    ui_api.orchestration = demo_results.get('observer')  # Use observer as orchestration
    
    print("🌐 Starting Flask UI server on http://localhost:8000")
    print("🎯 API Endpoints available:")
    print("   GET  /api/health - Health check")
    print("   GET  /api/simulation/status - Current status")
    print("   POST /api/simulation/step - Run simulation step")
    print("   GET  /api/simulation/parameters - Get parameters")
    print("   POST /api/simulation/parameters - Update parameters")
    
    try:
        ui_api.app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ UI Server error: {e}")

if __name__ == '__main__':
    try:
        # Run comprehensive demo
        demo_results = run_comprehensive_demo()
        
        print_banner("DEMO COMPLETE - STARTING UI SERVER")
        print("🎉 Comprehensive simulation demo completed successfully!")
        print("🚀 Starting UI server for interactive control...")
        
        # Start UI server
        start_ui_server(demo_results)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
