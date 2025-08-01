#!/usr/bin/env python3
"""
Test hybrid modeling integration (ABM + DES + SD coupling)
Validates Step 5 of the optimization checklist
"""

from economy_rewards import SDSystem
from economy_sim import MesaBakeryModel, StochasticEventSystem
import simpy
import time

def test_abm_sd_coupling():
    """Test ABM to SD coupling with cash flows affecting agents"""
    print("🧪 Testing ABM-SD Coupling...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Simulate ABM agent data
    abm_data = {
        "customers": 0.75,  # 75% customer satisfaction
        "labor": 0.80,      # 80% labor efficiency
        "suppliers": 0.70,  # 70% supplier reliability
        "partners": 0.85    # 85% partner engagement
    }
    
    # Test ABM coupling update
    cash_flows = sd_system.update_abm_coupling(abm_data)
    
    print(f"   ABM agent types: {len(abm_data)}")
    print(f"   Cash flow updates: {len(cash_flows)}")
    
    # Check cash flow distribution
    expected_flows = ["customers", "labor", "suppliers"]
    flows_generated = all(flow_type in cash_flows for flow_type in expected_flows)
    
    # Check agent performance tracking
    performance_tracked = len(sd_system.abm_coupling["agent_performance"]) == len(abm_data)
    
    print(f"   Cash flows generated: {flows_generated}")
    print(f"   Agent performance tracked: {performance_tracked}")
    
    # Verify cash flow values are reasonable
    reasonable_flows = all(0 <= flow <= 100 for flow in cash_flows.values())
    
    print(f"   Cash flow values reasonable: {reasonable_flows}")
    
    success = flows_generated and performance_tracked and reasonable_flows
    print(f"   ✅ ABM-SD coupling: {success}")
    
    return success

def test_des_sd_coupling():
    """Test DES to SD coupling with milling delays affecting inventory"""
    print("\n🧪 Testing DES-SD Coupling...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Simulate DES events
    des_events = {
        "milling_delays": 3.5,  # 3.5 hours of milling delays
        "equipment_failures": [
            {"equipment": "oven", "repair_hours": 2.0, "downtime_cost": 100},
            {"equipment": "mixer", "repair_hours": 1.5, "downtime_cost": 75}
        ],
        "weather_delays": [
            {"time": 10, "delay_hours": 1.2, "impact": "delivery_delay"}
        ]
    }
    
    # Test DES coupling update
    inventory_impacts = sd_system.update_des_coupling(des_events)
    
    print(f"   DES event types: {len(des_events)}")
    print(f"   Inventory impacts: {len(inventory_impacts)}")
    
    # Check inventory impact types
    expected_impacts = ["flour", "bread", "delivery_efficiency"]
    impacts_generated = all(impact_type in inventory_impacts for impact_type in expected_impacts)
    
    # Check milling delay processing
    milling_delay_processed = sd_system.des_coupling["milling_delays"] == 3.5
    
    # Check equipment downtime calculation
    expected_downtime = 2.0 + 1.5  # Sum of repair hours
    downtime_calculated = sd_system.des_coupling["equipment_downtime"] == expected_downtime
    
    print(f"   Inventory impacts generated: {impacts_generated}")
    print(f"   Milling delays processed: {milling_delay_processed}")
    print(f"   Equipment downtime calculated: {downtime_calculated}")
    
    # Check SD system state updates
    flour_inventory_updated = "flour_inventory" in sd_system.system_state
    supply_capacity_updated = sd_system.system_state.get("supply_capacity", 1.0) < 1.0
    
    print(f"   Flour inventory updated: {flour_inventory_updated}")
    print(f"   Supply capacity reduced: {supply_capacity_updated}")
    
    success = (
        impacts_generated and
        milling_delay_processed and
        downtime_calculated and
        flour_inventory_updated and
        supply_capacity_updated
    )
    
    print(f"   ✅ DES-SD coupling: {success}")
    
    return success

def test_hybrid_step_integration():
    """Test integrated step method with ABM+DES+SD coupling"""
    print("\n🧪 Testing Hybrid Step Integration...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Prepare test data
    abm_data = {
        "customers": 0.78,
        "labor": 0.82,
        "suppliers": 0.75,
        "partners": 0.80
    }
    
    des_events = {
        "milling_delays": 2.0,
        "equipment_failures": [{"equipment": "oven", "repair_hours": 1.0, "downtime_cost": 50}],
        "weather_delays": []
    }
    
    # Run hybrid step
    step_results = sd_system.step(abm_data, des_events)
    
    print(f"   Step results keys: {list(step_results.keys())}")
    
    # Check step results structure
    has_coupling_results = "coupling_results" in step_results
    has_coupling_status = "coupling_status" in step_results
    has_system_state = "system_state" in step_results
    has_total_flow = "total_flow" in step_results
    
    print(f"   Coupling results: {has_coupling_results}")
    print(f"   Coupling status: {has_coupling_status}")
    print(f"   System state: {has_system_state}")
    print(f"   Total flow: {has_total_flow}")
    
    # Check coupling status details
    if has_coupling_status:
        coupling_status = step_results["coupling_status"]
        abm_coupling_active = coupling_status["abm_coupling"]["active_cash_flows"] > 0
        des_coupling_active = coupling_status["des_coupling"]["inventory_impacts"] > 0
        integration_healthy = coupling_status["integration_health"]["abm_sd_sync"]
        
        print(f"   ABM coupling active: {abm_coupling_active}")
        print(f"   DES coupling active: {des_coupling_active}")
        print(f"   Integration healthy: {integration_healthy}")
    else:
        abm_coupling_active = des_coupling_active = integration_healthy = False
    
    # Check system state updates from coupling
    system_state = step_results.get("system_state", {})
    community_impact_updated = system_state.get("community_impact", 0) > 0
    supply_capacity_affected = system_state.get("supply_capacity", 1.0) < 1.0
    
    print(f"   Community impact updated: {community_impact_updated}")
    print(f"   Supply capacity affected: {supply_capacity_affected}")
    
    success = (
        has_coupling_results and
        has_coupling_status and
        has_system_state and
        abm_coupling_active and
        des_coupling_active and
        integration_healthy and
        community_impact_updated
    )
    
    print(f"   ✅ Hybrid step integration: {success}")
    
    return success

def test_full_model_integration():
    """Test full model integration with Mesa ABM + SimPy DES + SD"""
    print("\n🧪 Testing Full Model Integration...")
    
    # Initialize all systems
    mesa_model = MesaBakeryModel(num_customers=20, num_bakers=5)
    sd_system = SDSystem()
    
    # Create SimPy environment for DES
    simpy_env = simpy.Environment()
    stochastic_events = StochasticEventSystem(simpy_env)
    
    # Run integrated simulation steps
    integration_results = []
    
    for step in range(3):  # Run 3 integration steps
        # Step 1: Run Mesa ABM
        mesa_model.step()
        
        # Step 2: Advance SimPy DES
        simpy_env.run(until=simpy_env.now + 1)
        
        # Step 3: Collect ABM data
        abm_data = {
            "customers": len(mesa_model.customer_agents) / 50,  # Normalized
            "labor": len([a for a in mesa_model.agents if hasattr(a, 'daily_wage')]) / 20,
            "suppliers": 0.75,  # Simulated
            "partners": 0.80    # Simulated
        }
        
        # Step 4: Collect DES events
        des_events = stochastic_events.get_recent_events(24.0)
        
        # Step 5: Run SD system with hybrid coupling
        sd_results = sd_system.step(abm_data, des_events)
        
        integration_results.append({
            "step": step,
            "abm_agents": len(mesa_model.agents),
            "des_events": sum(len(events) for events in des_events.values()),
            "sd_flows": len(sd_results.get("coupling_results", {}).get("abm_cash_flows", {})),
            "integration_health": sd_results.get("coupling_status", {}).get("integration_health", {})
        })
    
    print(f"   Integration steps completed: {len(integration_results)}")
    
    # Analyze integration results
    total_agents = sum(result["abm_agents"] for result in integration_results)
    total_events = sum(result["des_events"] for result in integration_results)
    total_flows = sum(result["sd_flows"] for result in integration_results)
    
    print(f"   Total ABM agents processed: {total_agents}")
    print(f"   Total DES events processed: {total_events}")
    print(f"   Total SD cash flows: {total_flows}")
    
    # Check integration health across steps
    healthy_integrations = sum(1 for result in integration_results 
                             if result["integration_health"].get("abm_sd_sync", False))
    
    print(f"   Healthy integrations: {healthy_integrations}/{len(integration_results)}")
    
    success = (
        len(integration_results) == 3 and
        total_agents > 0 and
        total_flows > 0 and
        healthy_integrations >= 2  # At least 2/3 healthy
    )
    
    print(f"   ✅ Full model integration: {success}")
    
    return success

def main():
    """Run all hybrid modeling tests"""
    print("🚀 HYBRID MODELING INTEGRATION TESTING")
    print("="*60)
    
    results = []
    
    # Test individual components
    results.append(test_abm_sd_coupling())
    results.append(test_des_sd_coupling())
    results.append(test_hybrid_step_integration())
    results.append(test_full_model_integration())
    
    # Overall results
    print("\n📊 TEST RESULTS SUMMARY:")
    print("="*60)
    
    test_names = [
        "ABM-SD Coupling",
        "DES-SD Coupling", 
        "Hybrid Step Integration",
        "Full Model Integration"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {name}: {status}")
    
    overall_success = all(results)
    fitness_impact = 0.90 if overall_success else 0.60
    
    print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    print(f"   Fitness Impact: {fitness_impact:.2f}")
    
    if overall_success:
        print("   SD: Hybrid integration added. Flows cash to agents. Delays milling to inventory. Fitness impact: 0.90")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
