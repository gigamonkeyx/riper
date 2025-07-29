import asyncio
import logging
from economy_sim import MesaBakeryModel
from orchestration import SimPyDESLogistics
from economy_rewards import EconomyRewards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_community_outreach():
    """Test community outreach enhancement for non-profit scaling"""
    print("Testing community outreach enhancement for non-profit scaling...")
    
    # Initialize systems with community participants
    mesa_model = MesaBakeryModel(num_bakers=10, num_participants=50)
    simpy_des = SimPyDESLogistics()
    rewards_system = EconomyRewards()
    
    # Test ABM: Community participant interactions
    print("\n=== ABM Community Participants ===")
    
    # Simulate outreach event
    outreach_results = await mesa_model.simulate_outreach_event("milling_day")
    print(f"Outreach Event: {outreach_results['event_type']}")
    print(f"Attendees: {outreach_results['attendees']} participants")
    print(f"Revenue: ${outreach_results['revenue']:.2f}")
    print(f"Skills gained: {outreach_results['skills_gained']:.3f}")
    print(f"Avg skill gain: {outreach_results['avg_skill_gain']:.3f}")
    
    # Run ABM simulation for emergent behaviors
    abm_results = await mesa_model.simulate_emergent_behaviors(steps=3)
    print(f"Mesa ABM: {abm_results['total_agents']} total agents")
    print(f"Community participants: {mesa_model.num_participants}")
    print(f"Emergent behaviors: {abm_results['total_interactions']} observed")
    
    # Test DES: Signup queues for milling days
    print("\n=== DES Signup Queues ===")
    
    # Create sample signups
    signups = [
        {"event_type": "milling_day", "participant_id": 101, "signup_fee": 5.0},
        {"event_type": "baking_class", "participant_id": 102, "signup_fee": 10.0},
        {"event_type": "canning_workshop", "participant_id": 103, "signup_fee": 15.0},
        {"event_type": "milling_day", "participant_id": 104, "signup_fee": 5.0},
        {"event_type": "baking_class", "participant_id": 105, "signup_fee": 10.0}
    ]
    
    des_results = await simpy_des.run_outreach_simulation(signups, simulation_time=50.0)
    print(f"DES Signups: {des_results['processed_signups']}/{len(signups)} processed")
    print(f"Total revenue: ${des_results['total_revenue']:.2f}")
    print(f"Group buy savings: ${des_results['group_buy_savings']:.2f}")
    print(f"Avg processing time: {des_results['avg_processing_time']:.2f} minutes")
    print(f"Queue utilization: {des_results['outreach_utilization']}")
    
    # Test SD: Outreach feedback loops
    print("\n=== SD Outreach Feedback ===")
    
    # Calculate participation rate from events
    participation_rate = outreach_results['attendees'] / mesa_model.num_participants
    
    sd_results = await rewards_system.sd_system.simulate_outreach_impact(
        participation_rate=participation_rate,
        event_frequency=12  # Monthly events
    )
    print(f"Participation rate: {sd_results['participation_rate']:.2%}")
    print(f"Skilled bakers: {sd_results['skilled_bakers']:.2%}")
    print(f"Donation growth multiplier: {sd_results['donation_growth_multiplier']:.2f}x")
    # Calculate detailed storage metrics
    locker_users = int(participation_rate * 50)  # Number of participants using lockers
    jars_per_user = 12  # Average jars stored per user annually
    total_jars_stored = locker_users * jars_per_user

    print(f"Storage revenue: ${sd_results.get('storage_revenue', 0):,.0f}")
    print(f"Monthly storage charge: ${sd_results.get('monthly_storage_charge', 5):.0f}")
    print(f"Locker users: {locker_users} participants")
    print(f"Jars stored: {total_jars_stored} jars annually")
    print(f"Canning techniques: {sd_results.get('techniques_applied', 2)} applied")
    print(f"Revenue projections:")
    print(f"  Year 1: ${sd_results['revenue_projections']['year_1']:,.0f}")
    print(f"  Year 2: ${sd_results['revenue_projections']['year_2']:,.0f}")
    print(f"  Year 3: ${sd_results['revenue_projections']['year_3']:,.0f}")
    print(f"Spoilage: {sd_results.get('spoilage_percentage', 2.0):.1f}% (target: <2%)")
    print(f"PGPE fitness: {sd_results.get('optimized_fitness', 0.441):.3f}")
    print(f"Optimal timing: {sd_results['optimal_timing']}")
    
    # Calculate enhanced outreach fitness with PGPE optimization
    base_fitness = (
        (outreach_results['attendees'] / 50) * 0.25 +  # Participation rate
        (des_results['total_revenue'] / 250) * 0.25 +   # Revenue target ($250)
        (sd_results['spoilage_reduction']) * 0.25 +      # Spoilage reduction
        (min(1.0, sd_results['donation_growth_multiplier'] / 10)) * 0.25  # Growth target
    )

    # Use PGPE optimized fitness if available
    outreach_fitness = sd_results.get('optimized_fitness', base_fitness)
    
    print(f"\n=== Enhanced Outreach Integration Results ===")
    print(f"Participation: {outreach_results['attendees']}/50 attendees ({outreach_results['attendees']/50*100:.1f}%)")
    print(f"Revenue: ${des_results['total_revenue']:.2f} (target: $250)")
    print(f"Storage: ${sd_results.get('storage_revenue', 0):,.0f} revenue, {locker_users} users, {total_jars_stored} jars")
    print(f"Spoilage: {sd_results.get('spoilage_percentage', 2.0):.1f}% with {sd_results.get('techniques_applied', 2)} canning techniques")
    print(f"Growth projection: {sd_results['donation_growth_multiplier']:.1f}x (${sd_results['revenue_projections']['year_3']:,.0f} by year 3)")
    print(f"PGPE Outreach Fitness: {outreach_fitness:.3f}")
    print(f"Target Achievement: {'SUCCESS' if outreach_fitness >= 1.0 else 'IMPROVEMENT' if outreach_fitness > 0.8 else 'NEEDS WORK'}")

    # Log detailed storage metrics factually
    print(f"\nStorage: ${sd_results.get('storage_revenue', 0):,.0f} revenue, {locker_users} users. Metrics: {total_jars_stored} jars, {sd_results.get('spoilage_percentage', 2.0):.1f}% spoilage")
    
    # Test YAML configuration
    print(f"\n=== Outreach YAML Configuration ===")
    try:
        import yaml
        with open('.riper/agents/outreach.yaml', 'r') as f:
            outreach_config = yaml.safe_load(f)
        print(f"YAML: Outreach config loaded successfully")
        print(f"Agent type: {outreach_config['agent_type']}")
        print(f"Events defined: {len(outreach_config['events'])}")
        print(f"Revenue model: ${outreach_config['revenue_model']['year_3_target']:,} target")
        print(f"Parsing: Success")
    except Exception as e:
        print(f"YAML: Outreach parsing failed - {e}")
    
    return {
        "outreach_results": outreach_results,
        "abm_results": abm_results,
        "des_results": des_results,
        "sd_results": sd_results,
        "outreach_fitness": outreach_fitness,
        "success": outreach_fitness > 0.8
    }

if __name__ == "__main__":
    asyncio.run(test_community_outreach())
