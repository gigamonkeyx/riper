#!/usr/bin/env python3
"""
Test Simulation with New Learnings:
- Batch production scaling (20x for bakers, 5x for interns)
- Lean staffing (5 bakers max, 1:1 ratio, 4 staff)
- Experienced bakers only
- Realistic productivity expectations
"""

from economy_sim import MesaBakeryModel
import time

def test_new_learnings():
    print('🚀 SIMULATION WITH NEW LEARNINGS')
    print('=' * 60)
    
    start_time = time.time()
    
    # Create model with proper lean staffing
    model = MesaBakeryModel(
        num_bakers=10,          # Will be capped at 5 by constraints
        num_participants=25,    # Community participants
        num_customers=50,       # Customer base
        num_labor=15,           # Will create: 5 bakers + 5 interns + 4 staff = 14 total
        num_suppliers=25,       # Supplier network
        num_partners=25,        # Community partners
        num_c_corps=2,          # B2B customers
        num_llcs=3,             # B2B customers
        num_gov_entities=2,     # Government entities
        width=20, height=20     # Reasonable grid size
    )
    
    setup_time = time.time() - start_time
    
    print(f'✅ Model Setup: {setup_time:.3f} seconds')
    print(f'✅ Total Agents: {len(model.agents)}')
    
    # Analyze labor structure
    bakers = [a for a in model.labor_agents if a.labor_type == "baker"]
    interns = [a for a in model.labor_agents if a.labor_type == "intern"]
    staff = [a for a in model.labor_agents if a.labor_type == "staff"]
    
    print()
    print('👥 LEAN LABOR STRUCTURE:')
    print(f'   Bakers: {len(bakers)} (max 5, experienced only)')
    print(f'   Interns: {len(interns)} (1:1 ratio with bakers)')
    print(f'   Staff: {len(staff)} (lean support roles)')
    print(f'   Total Labor: {len(model.labor_agents)} agents')
    
    # Run simulation steps
    print()
    print('⚡ RUNNING SIMULATION STEPS:')
    step_times = []
    for i in range(5):
        step_start = time.time()
        model.step()
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f'   Step {i+1}: {step_time:.3f}s')
    
    # Analyze production with batch scaling
    print()
    print('🍞 BATCH PRODUCTION ANALYSIS:')
    
    # Baker analysis
    baker_production = sum(a.bread_items_produced for a in bakers)
    baker_costs = sum(a.daily_wage_cost for a in bakers)
    if bakers:
        avg_baker_output = baker_production / len(bakers)
        avg_baker_cost = baker_costs / len(bakers)
        print(f'   Bakers: {baker_production} loaves ({avg_baker_output:.0f}/baker)')
        print(f'   Baker Costs: ${baker_costs:.2f} (${avg_baker_cost:.2f}/baker)')
    
    # Intern analysis
    intern_production = sum(a.bread_items_produced for a in interns)
    intern_costs = sum(a.daily_wage_cost for a in interns)
    if interns:
        avg_intern_output = intern_production / len(interns)
        avg_intern_cost = intern_costs / len(interns)
        print(f'   Interns: {intern_production} loaves ({avg_intern_output:.0f}/intern)')
        print(f'   Intern Costs: ${intern_costs:.2f} (${avg_intern_cost:.2f}/intern)')
    
    # Staff analysis
    staff_production = sum(a.bread_items_produced for a in staff)
    staff_costs = sum(a.daily_wage_cost for a in staff)
    if staff:
        avg_staff_output = staff_production / len(staff)
        avg_staff_cost = staff_costs / len(staff)
        print(f'   Staff: {staff_production} loaves ({avg_staff_output:.0f}/staff)')
        print(f'   Staff Costs: ${staff_costs:.2f} (${avg_staff_cost:.2f}/staff)')
    
    # Total analysis
    total_production = baker_production + intern_production + staff_production
    total_costs = baker_costs + intern_costs + staff_costs
    
    print()
    print('📊 TOTAL PERFORMANCE:')
    print(f'   Total Production: {total_production} loaves/day')
    print(f'   Total Labor Costs: ${total_costs:.2f}/day')
    print(f'   Cost per Loaf: ${total_costs/max(1, total_production):.2f}')
    
    # Revenue analysis
    bread_price = 5.0
    daily_revenue = total_production * bread_price
    labor_cost_percentage = (total_costs / daily_revenue) * 100 if daily_revenue > 0 else 0
    daily_profit = daily_revenue - total_costs
    
    print()
    print('💰 FINANCIAL ANALYSIS:')
    print(f'   Daily Revenue: ${daily_revenue:.2f} ({total_production} × ${bread_price})')
    print(f'   Labor Cost %: {labor_cost_percentage:.1f}% of revenue')
    print(f'   Daily Profit: ${daily_profit:.2f}')
    
    # Performance assessment
    if labor_cost_percentage < 40:
        financial_grade = '🟢 EXCELLENT'
    elif labor_cost_percentage < 60:
        financial_grade = '🟡 GOOD'
    elif labor_cost_percentage < 80:
        financial_grade = '🟠 ACCEPTABLE'
    else:
        financial_grade = '🔴 POOR'
    
    print(f'   Financial Grade: {financial_grade}')
    
    # Productivity comparison
    print()
    print('🎯 PRODUCTIVITY COMPARISON:')
    print('   OLD SYSTEM (without batch scaling):')
    print('     - 10 bakers: ~85 loaves/day')
    print('     - 20 interns: ~10 loaves/day')
    print('     - Total: ~95 loaves/day')
    print('     - Cost: $6,628/day')
    print('     - Profitability: TERRIBLE')
    
    print('   NEW SYSTEM (with batch scaling & lean staffing):')
    print(f'     - {len(bakers)} bakers: {baker_production} loaves/day')
    print(f'     - {len(interns)} interns: {intern_production} loaves/day')
    print(f'     - {len(staff)} staff: {staff_production} loaves/day')
    print(f'     - Total: {total_production} loaves/day')
    print(f'     - Cost: ${total_costs:.2f}/day')
    print(f'     - Profitability: {financial_grade}')
    
    improvement_factor = total_production / 95 if total_production > 0 else 0
    cost_reduction = (6628 - total_costs) / 6628 * 100 if total_costs < 6628 else 0
    
    print()
    print('🚀 IMPROVEMENT METRICS:')
    print(f'   Production Increase: {improvement_factor:.1f}x improvement')
    print(f'   Cost Reduction: {cost_reduction:.1f}% savings')
    print(f'   Efficiency Gain: {(improvement_factor * (1 + cost_reduction/100)):.1f}x overall')
    
    total_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)
    
    print()
    print('⚡ PERFORMANCE METRICS:')
    print(f'   Total Runtime: {total_time:.2f} seconds')
    print(f'   Avg Step Time: {avg_step_time:.3f} seconds')
    print(f'   Agents/Second: {len(model.agents)/avg_step_time:.0f}')
    
    print()
    print('🎉 NEW LEARNINGS TEST COMPLETE!')
    
    return {
        'total_production': total_production,
        'total_costs': total_costs,
        'daily_profit': daily_profit,
        'labor_cost_percentage': labor_cost_percentage,
        'financial_grade': financial_grade
    }

if __name__ == "__main__":
    results = test_new_learnings()
