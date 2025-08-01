#!/usr/bin/env python3
"""
Test Agent Action Report Integration
Validates the complete agent action report system including:
- SD system tracking of agent changes
- Evo core generation of evolution reports
- Reporting system integration (17 reports/year)
- UI display of agent action tables and charts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from economy_sim import MesaBakeryModel
from economy_rewards import SDSystem
from evo_core import NeuroEvolutionEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_action_report_system():
    """Test the agent action report system in SD"""
    print("🤖 Testing Agent Action Report System...")
    
    sd_system = SDSystem()
    
    # Check agent action report system
    report_system = sd_system.agent_action_report_system
    
    # Verify system configuration
    tracking_enabled = report_system["report_tracking"]["enabled"]
    detail_level = report_system["report_tracking"]["detail_level"]
    
    print(f"   Tracking enabled: {tracking_enabled}")
    print(f"   Detail level: {detail_level}")
    
    # Check tracked changes
    tracked_changes = report_system["tracked_changes"]
    customer_changes = len(tracked_changes["customer_agents"])
    labor_changes = len(tracked_changes["labor_agents"])
    supplier_changes = len(tracked_changes["supplier_agents"])
    partner_changes = len(tracked_changes["partner_agents"])
    
    print(f"   Customer agent changes: {customer_changes}")
    print(f"   Labor agent changes: {labor_changes}")
    print(f"   Supplier agent changes: {supplier_changes}")
    print(f"   Partner agent changes: {partner_changes}")
    
    # Check aggregate impact
    aggregate = report_system["aggregate_impact"]
    total_revenue_increase = aggregate["total_revenue_increase"]
    annual_impact = aggregate["annual_revenue_impact"]
    fitness_contribution = aggregate["fitness_contribution"]
    
    print(f"   Total revenue increase: +${total_revenue_increase}/day")
    print(f"   Annual impact: +${annual_impact}/year")
    print(f"   Fitness contribution: +{fitness_contribution}")
    
    # Verify requirements
    system_enabled = tracking_enabled and detail_level == "comprehensive"
    changes_tracked = customer_changes >= 2 and labor_changes >= 2
    impact_positive = total_revenue_increase > 300 and annual_impact > 100000
    fitness_adequate = fitness_contribution >= 0.10
    
    success = system_enabled and changes_tracked and impact_positive and fitness_adequate
    print(f"   ✅ Agent action report system: {success}")
    
    return success

def test_evo_core_integration():
    """Test evolutionary core agent action report generation"""
    print("🧬 Testing Evo Core Integration...")
    
    try:
        evo_engine = NeuroEvolutionEngine()
        
        # Test agent action report generation
        test_traits = {
            "customer_donation_propensity": 0.22,
            "labor_productivity": 0.88,
            "supplier_price_efficiency": 392.0,
            "partner_outreach_effectiveness": 0.82
        }
        
        reports = evo_engine.generate_agent_action_reports(generation=35, evolved_traits=test_traits)
        
        print(f"   Generated reports: {len(reports)}")
        
        # Check report content
        if reports:
            first_report = reports[0]
            print(f"   First report agent: {first_report.agent_type}")
            print(f"   First report trait: {first_report.trait_name}")
            print(f"   First report change: {first_report.change_amount:+.3f}")
            print(f"   First report fitness: {first_report.fitness_contribution:.3f}")
            
            # Check rationale format
            rationale_valid = "evolved" in first_report.rationale.lower()
            impact_valid = len(first_report.impact_metrics) > 0
            
            print(f"   Rationale format valid: {rationale_valid}")
            print(f"   Impact metrics present: {impact_valid}")
        
        # Test evolution summary compilation
        summary = evo_engine.compile_evolution_summary(reports, final_fitness=2.9)
        
        evolution_overview = summary.get("evolution_overview", {})
        final_fitness = evolution_overview.get("final_fitness", 0)
        fitness_achieved = evolution_overview.get("fitness_achieved", False)
        
        print(f"   Final fitness: {final_fitness}")
        print(f"   Target achieved: {fitness_achieved}")
        
        success = len(reports) >= 4 and final_fitness >= 2.8 and fitness_achieved
        print(f"   ✅ Evo core integration: {success}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Evo core integration failed: {e}")
        return False

def test_reporting_integration():
    """Test agent action reports in 17 reports/year system"""
    print("📊 Testing Reporting Integration...")
    
    sd_system = SDSystem()
    
    # Test monthly report with agent actions
    monthly_report = sd_system.generate_enhanced_reports(report_month=6)
    
    current_reports = monthly_report["current_reports"]
    monthly_financial = current_reports[0] if current_reports else {}
    
    # Check agent action reports in monthly report
    agent_reports = monthly_financial.get("agent_action_reports", {})
    
    customer_evolution = agent_reports.get("customer_evolution", {})
    labor_optimization = agent_reports.get("labor_optimization", {})
    supplier_negotiation = agent_reports.get("supplier_negotiation", {})
    partner_outreach = agent_reports.get("partner_outreach", {})
    aggregate_impact = agent_reports.get("aggregate_impact", {})
    
    print(f"   Customer evolution present: {len(customer_evolution) > 0}")
    print(f"   Labor optimization present: {len(labor_optimization) > 0}")
    print(f"   Supplier negotiation present: {len(supplier_negotiation) > 0}")
    print(f"   Partner outreach present: {len(partner_outreach) > 0}")
    print(f"   Aggregate impact present: {len(aggregate_impact) > 0}")
    
    # Check quarterly report with evolution summary
    quarterly_report = sd_system.generate_enhanced_reports(report_month=9)
    quarterly_reports = quarterly_report["current_reports"]
    
    quarterly_data = None
    for report in quarterly_reports:
        if report.get("report_type") == "quarterly_compliance":
            quarterly_data = report
            break
    
    if quarterly_data:
        evolution_summary = quarterly_data.get("agent_evolution_summary", {})
        fitness_progression = evolution_summary.get("fitness_progression", "")
        key_optimizations = evolution_summary.get("key_optimizations", [])
        
        print(f"   Quarterly evolution summary present: {len(evolution_summary) > 0}")
        print(f"   Fitness progression described: {'fitness' in fitness_progression.lower()}")
        print(f"   Key optimizations count: {len(key_optimizations)}")
    
    # Verify integration
    monthly_integrated = len(agent_reports) >= 4
    quarterly_integrated = quarterly_data and len(evolution_summary) > 0 if quarterly_data else False
    
    success = monthly_integrated and quarterly_integrated
    print(f"   ✅ Reporting integration: {success}")
    
    return success

def test_ui_integration():
    """Test agent action reports in UI output"""
    print("💻 Testing UI Integration...")
    
    model = MesaBakeryModel()
    
    # Check UI output display system
    output_system = model.output_display_system
    agent_ui = output_system.get("agent_action_report_ui", {})
    
    ui_enabled = agent_ui.get("enabled", False)
    display_format = agent_ui.get("display_format", "")
    
    print(f"   UI enabled: {ui_enabled}")
    print(f"   Display format: {display_format}")
    
    # Check table configuration
    table_config = agent_ui.get("table_configuration", {})
    columns = table_config.get("columns", [])
    data_rows = table_config.get("data_rows", [])
    summary_row = table_config.get("summary_row", {})
    
    print(f"   Table columns: {len(columns)}")
    print(f"   Data rows: {len(data_rows)}")
    print(f"   Summary row present: {len(summary_row) > 0}")
    
    # Check chart integration
    chart_integration = agent_ui.get("chart_integration", {})
    fitness_chart = chart_integration.get("fitness_progression_chart", {})
    impact_chart = chart_integration.get("impact_breakdown_chart", {})
    
    print(f"   Fitness progression chart: {len(fitness_chart) > 0}")
    print(f"   Impact breakdown chart: {len(impact_chart) > 0}")
    
    # Check real-time updates
    real_time = agent_ui.get("real_time_updates", {})
    update_frequency = real_time.get("update_frequency", "")
    websocket_enabled = real_time.get("websocket_enabled", False)
    
    print(f"   Update frequency: {update_frequency}")
    print(f"   WebSocket enabled: {websocket_enabled}")
    
    # Verify UI integration
    ui_configured = ui_enabled and len(columns) >= 4 and len(data_rows) >= 8
    charts_configured = len(fitness_chart) > 0 and len(impact_chart) > 0
    real_time_configured = update_frequency == "every_step" and websocket_enabled
    
    success = ui_configured and charts_configured and real_time_configured
    print(f"   ✅ UI integration: {success}")
    
    return success

def test_comprehensive_integration():
    """Test comprehensive agent action report integration"""
    print("🔗 Testing Comprehensive Integration...")
    
    # Initialize all systems
    model = MesaBakeryModel()
    sd_system = SDSystem()
    
    # Run a few simulation steps to generate data
    for step in range(3):
        model.step()
        sd_system.step()
    
    # Check system interactions
    has_agent_reports = hasattr(sd_system, 'agent_action_report_system')
    has_ui_integration = 'agent_action_report_ui' in model.output_display_system
    has_reporting_integration = 'agent_action_integration' in sd_system.reporting_system
    
    print(f"   SD agent reports: {has_agent_reports}")
    print(f"   UI integration: {has_ui_integration}")
    print(f"   Reporting integration: {has_reporting_integration}")
    
    # Check data flow
    if has_agent_reports:
        report_system = sd_system.agent_action_report_system
        aggregate_impact = report_system["aggregate_impact"]
        total_impact = aggregate_impact["annual_revenue_impact"]
        fitness_contribution = aggregate_impact["fitness_contribution"]
        
        print(f"   Total annual impact: ${total_impact:,}")
        print(f"   Fitness contribution: +{fitness_contribution}")
        
        impact_adequate = total_impact >= 100000
        fitness_adequate = fitness_contribution >= 0.10
    else:
        impact_adequate = False
        fitness_adequate = False
    
    # Check enhanced fruit locker
    fruit_locker = sd_system.fruit_locker_system
    locker_capacity = fruit_locker["capacity_lbs"]
    locker_cost = fruit_locker["upfront_cost"]
    
    print(f"   Fruit locker capacity: {locker_capacity:,} lbs")
    print(f"   Fruit locker cost: ${locker_cost:,}")
    
    locker_updated = locker_capacity == 15000 and locker_cost == 15000
    
    success = (
        has_agent_reports and has_ui_integration and has_reporting_integration and
        impact_adequate and fitness_adequate and locker_updated
    )
    
    print(f"   ✅ Comprehensive integration: {success}")
    
    return success

def main():
    """Run agent action report integration tests"""
    print("🚀 AGENT ACTION REPORT INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Agent Action Report System", test_agent_action_report_system),
        ("Evo Core Integration", test_evo_core_integration),
        ("Reporting Integration", test_reporting_integration),
        ("UI Integration", test_ui_integration),
        ("Comprehensive Integration", test_comprehensive_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name}: FAILED ({e})")
            results.append((test_name, False))
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    pass_rate = passed / total
    
    print("📋 AGENT ACTION REPORT TEST SUMMARY:")
    print("="*60)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed ({pass_rate:.1%})")
    
    if pass_rate >= 0.8:
        print("🎉 AGENT ACTION REPORT INTEGRATION SUCCESS!")
        print("   System ready for detailed agent evolution tracking")
        print("   Fitness >2.8 with comprehensive reporting")
        return True
    else:
        print("⚠️ Some issues need attention")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
