import asyncio
import logging
from economy_sim import EconomySimulator, SimConfig, ABMSystem
from orchestration import DESLogisticsSystem
from economy_rewards import EconomyRewards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_abm_des_sd_integration():
    """Test ABM/DES/SD integration for Tonasket sim module"""
    print("Testing ABM/DES/SD integration for Tonasket sim...")
    
    # Initialize systems
    config = SimConfig(years=1, initial_funding=150000.0)
    economy_sim = EconomySimulator(config)
    des_system = DESLogisticsSystem()
    rewards_system = EconomyRewards()
    
    # Test ABM: Baker labor agents
    print("\n=== ABM Testing ===")
    abm_results = await economy_sim.grant_model.abm_system.simulate_emergent_behaviors()
    print(f"ABM: Agents {abm_results['total_agents']} defined. Emergent behaviors: {abm_results['total_interactions']} observed.")
    print(f"Fitness impact: Skill level {abm_results['avg_skill_level']:.3f}, Productivity {abm_results['avg_productivity']:.3f}")
    
    # Test DES: Grain donation logistics
    print("\n=== DES Testing ===")
    # Add sample donation events
    des_system.add_donation_event("Bluebird Farm", "Pie Factory", 500.0, 0.0)
    des_system.add_donation_event("Okanogan Valley Grains", "Food Bank", 800.0, 1.0)
    des_system.add_donation_event("Community Gardens", "School Kitchen", 200.0, 2.0)
    
    des_results = await des_system.process_donation_queue(max_events=5)
    print(f"DES: Queues {des_results['processed_events']}/3 processed. Throughput: {des_results['total_units']:.1f} units.")
    print(f"Perf: {des_results['avg_processing_time']:.2f} seconds avg, Efficiency: {des_results['queue_efficiency']:.3f}")
    
    # Test SD: Feedback loops
    print("\n=== SD Testing ===")
    sd_results = await rewards_system.sd_system.simulate_feedback_dynamics(
        grant_change=0.2,  # 20% increase in grants
        demand_change=0.1   # 10% increase in demand
    )
    print(f"SD: Loops {sd_results['active_loops']} active. System change: {sd_results['total_system_change']:.3f}")
    print(f"Stability: {sd_results['stability_score']:.3f}")
    
    # Test integrated simulation
    print("\n=== Integrated Simulation ===")
    sim_results = await economy_sim.grant_model.simulate_year(1)
    
    # Calculate overall fitness
    integration_fitness = (
        abm_results['avg_productivity'] * 0.3 +
        des_results['queue_efficiency'] * 0.3 +
        sd_results['stability_score'] * 0.4
    )
    
    print(f"\n=== Results Summary ===")
    print(f"ABM Agents: {abm_results['total_agents']}")
    print(f"DES Throughput: {des_results['total_units']:.1f} units")
    print(f"SD Stability: {sd_results['stability_score']:.3f}")
    print(f"Integration Fitness: {integration_fitness:.3f}")
    print(f"Target Achievement: {'SUCCESS' if integration_fitness > 0.8 else 'NEEDS IMPROVEMENT'}")
    
    # Test YAML configuration parsing
    print(f"\n=== YAML Configuration ===")
    try:
        import yaml
        with open('.riper/agents/logistics.yaml', 'r') as f:
            logistics_config = yaml.safe_load(f)
        print(f"YAML: Config loaded successfully. Agent type: {logistics_config['agent_type']}")
        print(f"Tasks defined: {len(logistics_config['tasks'])}")
        print(f"Parsing: Success")
    except Exception as e:
        print(f"YAML: Parsing failed - {e}")
    
    return {
        "abm_results": abm_results,
        "des_results": des_results,
        "sd_results": sd_results,
        "integration_fitness": integration_fitness,
        "success": integration_fitness > 0.8
    }

if __name__ == "__main__":
    asyncio.run(test_abm_des_sd_integration())
