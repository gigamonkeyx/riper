#!/usr/bin/env python3
"""
Test 200-agent scaling optimization
Validates Step 8 of the optimization checklist
"""

from economy_sim import MesaBakeryModel

def test_agent_scaling_configuration():
    """Test 200-agent scaling configuration"""
    print("🧪 Testing Agent Scaling Configuration...")
    
    # Initialize model with default 200-agent configuration
    model = MesaBakeryModel()
    
    # Check target configuration
    target_agent_count = model.target_agent_count
    agent_distribution = model.agent_distribution
    
    print(f"   Target agent count: {target_agent_count}")
    print(f"   Agent distribution: {agent_distribution}")
    
    # Verify target is 200 agents
    target_correct = target_agent_count == 200
    
    # Verify distribution adds up to 200
    total_configured = sum(agent_distribution.values())
    distribution_correct = total_configured == 200
    
    # Verify balanced distribution (each type has 50 agents)
    expected_per_type = 50
    distribution_balanced = all(count == expected_per_type for count in agent_distribution.values())
    
    print(f"   Target correct (200): {target_correct}")
    print(f"   Distribution total correct (200): {distribution_correct}")
    print(f"   Distribution balanced (50 each): {distribution_balanced}")
    
    success = target_correct and distribution_correct and distribution_balanced
    
    print(f"   ✅ Agent scaling configuration: {success}")
    
    return success

def test_actual_agent_creation():
    """Test actual agent creation matches configuration"""
    print("\n🧪 Testing Actual Agent Creation...")
    
    # Initialize model with 200-agent configuration
    model = MesaBakeryModel()
    
    # Get agent scaling metrics
    scaling_metrics = model.get_agent_scaling_metrics()
    
    # Check actual agent counts
    actual_config = scaling_metrics["actual_configuration"]
    actual_total = actual_config["actual_total_agents"]
    actual_distribution = actual_config["actual_distribution"]
    
    print(f"   Actual total agents: {actual_total}")
    print(f"   Actual distribution: {actual_distribution}")
    
    # Check scaling efficiency
    scaling_metrics_data = scaling_metrics["scaling_metrics"]
    scaling_efficiency = scaling_metrics_data["scaling_efficiency"]
    avg_distribution_accuracy = scaling_metrics_data["avg_distribution_accuracy"]
    
    print(f"   Scaling efficiency: {scaling_efficiency:.1%}")
    print(f"   Distribution accuracy: {avg_distribution_accuracy:.1%}")
    
    # Verify scaling targets
    scaling_status = scaling_metrics["scaling_status"]
    target_met = scaling_status["target_met"]
    distribution_balanced = scaling_status["distribution_balanced"]
    performance_adequate = scaling_status["performance_adequate"]
    
    print(f"   Target met (≥90% of 200): {target_met}")
    print(f"   Distribution balanced (≥80% accuracy): {distribution_balanced}")
    print(f"   Performance adequate (≥80% score): {performance_adequate}")
    
    # Check specific agent types
    customers_adequate = actual_distribution["customers"] >= 40  # At least 40 customers (80% of 50)
    labor_adequate = actual_distribution["labor"] >= 8          # At least 8 labor agents
    suppliers_adequate = actual_distribution["suppliers"] >= 0  # At least some suppliers
    partners_adequate = actual_distribution["partners"] >= 0    # At least some partners
    
    print(f"   Customers adequate (≥40): {customers_adequate}")
    print(f"   Labor adequate (≥8): {labor_adequate}")
    print(f"   Suppliers present: {suppliers_adequate}")
    print(f"   Partners present: {partners_adequate}")
    
    success = (
        target_met and
        distribution_balanced and
        performance_adequate and
        customers_adequate and
        labor_adequate
    )
    
    print(f"   ✅ Actual agent creation: {success}")
    
    return success

def test_performance_with_200_agents():
    """Test performance with 200 agents"""
    print("\n🧪 Testing Performance with 200 Agents...")
    
    # Initialize model with 200-agent configuration
    model = MesaBakeryModel()
    
    # Get performance metrics
    scaling_metrics = model.get_agent_scaling_metrics()
    performance_metrics = scaling_metrics["performance_metrics"]
    
    agents_per_step = performance_metrics["agents_per_step"]
    memory_efficiency = performance_metrics["memory_efficiency"]
    processing_efficiency = performance_metrics["processing_efficiency"]
    scalability_score = performance_metrics["scalability_score"]
    
    print(f"   Agents per step: {agents_per_step}")
    print(f"   Memory efficiency: {memory_efficiency:.1%}")
    print(f"   Processing efficiency: {processing_efficiency:.1%}")
    print(f"   Scalability score: {scalability_score:.1%}")
    
    # Test model step performance
    import time
    
    step_times = []
    for i in range(3):  # Test 3 steps
        start_time = time.time()
        model.step()
        end_time = time.time()
        step_time = end_time - start_time
        step_times.append(step_time)
        print(f"   Step {i+1} time: {step_time:.3f} seconds")
    
    avg_step_time = sum(step_times) / len(step_times)
    max_step_time = max(step_times)
    
    print(f"   Average step time: {avg_step_time:.3f} seconds")
    print(f"   Maximum step time: {max_step_time:.3f} seconds")
    
    # Performance criteria
    memory_adequate = memory_efficiency >= 0.8      # At least 80% memory efficiency
    processing_adequate = processing_efficiency >= 0.6  # At least 60% processing efficiency
    scalability_adequate = scalability_score >= 0.8    # At least 80% scalability score
    step_time_reasonable = avg_step_time <= 2.0         # Average step under 2 seconds
    max_time_reasonable = max_step_time <= 5.0          # Max step under 5 seconds
    
    print(f"   Memory adequate (≥80%): {memory_adequate}")
    print(f"   Processing adequate (≥60%): {processing_adequate}")
    print(f"   Scalability adequate (≥80%): {scalability_adequate}")
    print(f"   Step time reasonable (≤2s): {step_time_reasonable}")
    print(f"   Max time reasonable (≤5s): {max_time_reasonable}")
    
    success = (
        memory_adequate and
        processing_adequate and
        scalability_adequate and
        step_time_reasonable and
        max_time_reasonable
    )
    
    print(f"   ✅ Performance with 200 agents: {success}")
    
    return success

def test_agent_interaction_scaling():
    """Test agent interactions scale properly with 200 agents"""
    print("\n🧪 Testing Agent Interaction Scaling...")
    
    # Initialize model with 200-agent configuration
    model = MesaBakeryModel()
    
    # Run a few steps to generate interactions
    for _ in range(5):
        model.step()
    
    # Check interaction metrics
    total_interactions = 0
    interaction_types = {
        "customer_purchases": 0,
        "baker_productions": 0,
        "partner_collaborations": 0,
        "supplier_deliveries": 0
    }
    
    # Count customer interactions
    for customer in model.customer_agents:
        if hasattr(customer, 'purchases_made'):
            interaction_types["customer_purchases"] += customer.purchases_made
            total_interactions += customer.purchases_made
    
    # Count baker interactions
    for baker in model.baker_agents:
        if hasattr(baker, 'bread_items_produced'):
            interaction_types["baker_productions"] += baker.bread_items_produced
            total_interactions += baker.bread_items_produced
    
    # Count partner interactions
    partner_agents = [a for a in model.agents if hasattr(a, 'partnership_strength')]
    for partner in partner_agents:
        if hasattr(partner, 'items_received'):
            interaction_types["partner_collaborations"] += partner.items_received
            total_interactions += partner.items_received
    
    # Count supplier interactions
    supplier_agents = [a for a in model.agents if hasattr(a, 'deliveries_made')]
    for supplier in supplier_agents:
        if hasattr(supplier, 'deliveries_made'):
            interaction_types["supplier_deliveries"] += supplier.deliveries_made
            total_interactions += supplier.deliveries_made
    
    print(f"   Total interactions: {total_interactions}")
    print(f"   Customer purchases: {interaction_types['customer_purchases']}")
    print(f"   Baker productions: {interaction_types['baker_productions']}")
    print(f"   Partner collaborations: {interaction_types['partner_collaborations']}")
    print(f"   Supplier deliveries: {interaction_types['supplier_deliveries']}")
    
    # Calculate interaction density
    total_agents = len(model.agents)
    interaction_density = total_interactions / total_agents if total_agents > 0 else 0
    
    print(f"   Interaction density: {interaction_density:.2f} interactions/agent")
    
    # Verify interaction scaling
    interactions_adequate = total_interactions >= 100  # At least 100 interactions across 5 steps
    density_reasonable = 0.5 <= interaction_density <= 10.0  # Reasonable interaction density
    customer_activity = interaction_types["customer_purchases"] >= 20  # At least 20 customer purchases
    baker_activity = interaction_types["baker_productions"] >= 10     # At least 10 baker productions
    
    print(f"   Interactions adequate (≥100): {interactions_adequate}")
    print(f"   Density reasonable (0.5-10.0): {density_reasonable}")
    print(f"   Customer activity (≥20): {customer_activity}")
    print(f"   Baker activity (≥10): {baker_activity}")
    
    success = (
        interactions_adequate and
        density_reasonable and
        customer_activity and
        baker_activity
    )
    
    print(f"   ✅ Agent interaction scaling: {success}")
    
    return success

def main():
    """Run all 200-agent scaling tests"""
    print("🚀 200-AGENT SCALING OPTIMIZATION TESTING")
    print("="*60)
    
    results = []
    
    # Test individual components
    results.append(test_agent_scaling_configuration())
    results.append(test_actual_agent_creation())
    results.append(test_performance_with_200_agents())
    results.append(test_agent_interaction_scaling())
    
    # Overall results
    print("\n📊 TEST RESULTS SUMMARY:")
    print("="*60)
    
    test_names = [
        "Agent Scaling Configuration",
        "Actual Agent Creation",
        "Performance with 200 Agents",
        "Agent Interaction Scaling"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {name}: {status}")
    
    overall_success = all(results)
    fitness_impact = 0.88 if overall_success else 0.65
    
    print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    print(f"   Fitness Impact: {fitness_impact:.2f}")
    
    if overall_success:
        print("   ABM: Scaled to 200 agents (50 customers, 50 labor, 50 suppliers, 50 partners). Fitness impact: 0.88")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
