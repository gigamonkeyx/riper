#!/usr/bin/env python3
"""
Test Simulation with Wage Adjustments and Expanded Intern Responsibilities:
- Bakers: $25/hour (up from $20-25 range)
- Baker Interns: $17/hour (up from $15)
- Expanded intern duties: prep work, counter food, retail sales
- Higher intern productivity and batch sizes
"""

from economy_sim import MesaBakeryModel
import time

def test_wage_adjustments():
    print('💰 SIMULATION WITH WAGE ADJUSTMENTS & EXPANDED INTERN ROLES')
    print('=' * 70)
    
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
    print('👥 UPDATED LABOR STRUCTURE:')
    print(f'   Experienced Bakers: {len(bakers)} @ $25/hour')
    print(f'   Baker Interns: {len(interns)} @ $17/hour (expanded roles)')
    print(f'   Support Staff: {len(staff)} @ $16-22/hour')
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
    
    # Detailed analysis
    print()
    print('🍞 DETAILED PRODUCTION & WAGE ANALYSIS:')
    
    # Baker analysis
    baker_production = sum(a.bread_items_produced for a in bakers)
    baker_costs = sum(a.daily_wage_cost for a in bakers)
    if bakers:
        avg_baker_output = baker_production / len(bakers)
        avg_baker_cost = baker_costs / len(bakers)
        baker_hourly = bakers[0].hourly_wage
        baker_batch = bakers[0].batch_size
        print(f'   👨‍🍳 EXPERIENCED BAKERS:')
        print(f'      Production: {baker_production} loaves ({avg_baker_output:.0f}/baker)')
        print(f'      Wage: ${baker_hourly}/hour × 8 hours = ${baker_hourly * 8}/day')
        print(f'      Total Cost: ${baker_costs:.2f} (${avg_baker_cost:.2f}/baker)')
        print(f'      Batch Size: {baker_batch} loaves/unit')
        print(f'      Cost per Loaf: ${baker_costs/max(1, baker_production):.2f}')
    
    # Intern analysis with expanded responsibilities
    intern_production = sum(a.bread_items_produced for a in interns)
    intern_costs = sum(a.daily_wage_cost for a in interns)
    if interns:
        avg_intern_output = intern_production / len(interns)
        avg_intern_cost = intern_costs / len(interns)
        intern_hourly = interns[0].hourly_wage
        intern_batch = interns[0].batch_size
        
        # Calculate support value
        total_prep_value = sum(getattr(a, 'prep_work_value', 0) for a in interns)
        total_counter_value = sum(getattr(a, 'counter_food_value', 0) for a in interns)
        total_retail_value = sum(getattr(a, 'retail_sales_value', 0) for a in interns)
        total_support_value = total_prep_value + total_counter_value + total_retail_value
        
        print(f'   🎓 BAKER INTERNS (EXPANDED ROLES):')
        print(f'      Bread Production: {intern_production} loaves ({avg_intern_output:.0f}/intern)')
        print(f'      Wage: ${intern_hourly}/hour × 8 hours = ${intern_hourly * 8}/day')
        print(f'      Total Cost: ${intern_costs:.2f} (${avg_intern_cost:.2f}/intern)')
        print(f'      Batch Size: {intern_batch} loaves/unit (improved)')
        print(f'      Support Tasks Value: {total_support_value:.1f} units')
        print(f'        - Prep Work: {total_prep_value:.1f} units')
        print(f'        - Counter Food: {total_counter_value:.1f} units')
        print(f'        - Retail Sales: {total_retail_value:.1f} units')
        print(f'      Cost per Loaf: ${intern_costs/max(1, intern_production):.2f}')
    
    # Staff analysis
    staff_production = sum(a.bread_items_produced for a in staff)
    staff_costs = sum(a.daily_wage_cost for a in staff)
    if staff:
        avg_staff_output = staff_production / len(staff)
        avg_staff_cost = staff_costs / len(staff)
        print(f'   👥 SUPPORT STAFF:')
        print(f'      Production: {staff_production} loaves ({avg_staff_output:.0f}/staff)')
        print(f'      Total Cost: ${staff_costs:.2f} (${avg_staff_cost:.2f}/staff)')
    
    # Total analysis
    total_production = baker_production + intern_production + staff_production
    total_costs = baker_costs + intern_costs + staff_costs
    
    print()
    print('📊 TOTAL PERFORMANCE WITH WAGE ADJUSTMENTS:')
    print(f'   Total Production: {total_production} loaves/day')
    print(f'   Total Labor Costs: ${total_costs:.2f}/day')
    print(f'   Cost per Loaf: ${total_costs/max(1, total_production):.2f}')
    
    # Revenue analysis
    bread_price = 5.0
    daily_revenue = total_production * bread_price
    labor_cost_percentage = (total_costs / daily_revenue) * 100 if daily_revenue > 0 else 0
    daily_profit = daily_revenue - total_costs
    
    print()
    print('💰 FINANCIAL IMPACT OF WAGE INCREASES:')
    print(f'   Daily Revenue: ${daily_revenue:.2f} ({total_production} × ${bread_price})')
    print(f'   Labor Cost %: {labor_cost_percentage:.1f}% of revenue')
    print(f'   Daily Profit: ${daily_profit:.2f}')
    
    # Compare to previous version
    print()
    print('📈 COMPARISON TO PREVIOUS VERSION:')
    print('   PREVIOUS (lower wages):')
    print('     - Bakers: $20-25/hour → Production: ~902 loaves')
    print('     - Interns: $15/hour → Production: ~74 loaves')
    print('     - Total Cost: ~$2,148/day')
    print('     - Profit: ~$2,772/day')
    
    print('   CURRENT (adjusted wages + expanded roles):')
    print(f'     - Bakers: $25/hour → Production: {baker_production} loaves')
    print(f'     - Interns: $17/hour + support tasks → Production: {intern_production} loaves')
    print(f'     - Total Cost: ${total_costs:.2f}/day')
    print(f'     - Profit: ${daily_profit:.2f}/day')
    
    # Calculate changes
    cost_increase = total_costs - 2148
    profit_change = daily_profit - 2772
    
    print()
    print('💡 IMPACT ANALYSIS:')
    print(f'   Cost Increase: ${cost_increase:.2f}/day ({(cost_increase/2148)*100:.1f}%)')
    print(f'   Profit Change: ${profit_change:.2f}/day')
    
    if profit_change > 0:
        print('   ✅ Higher wages INCREASED profitability (better productivity)')
    elif profit_change > -200:
        print('   🟡 Higher wages slightly reduced profit (acceptable trade-off)')
    else:
        print('   🔴 Higher wages significantly reduced profitability')
    
    # ROI on wage increases
    if cost_increase > 0:
        productivity_gain = (total_production - 976) * bread_price  # Assuming previous was ~976 loaves
        roi = (productivity_gain - cost_increase) / cost_increase * 100
        print(f'   ROI on Wage Increase: {roi:.1f}%')
    
    print()
    print('🎯 INTERN VALUE PROPOSITION:')
    if interns:
        intern_total_value = intern_production * bread_price + (total_support_value * 2)  # Support tasks worth $2/unit
        intern_roi = (intern_total_value - intern_costs) / intern_costs * 100
        print(f'   Intern Total Value: ${intern_total_value:.2f}/day')
        print(f'   Intern ROI: {intern_roi:.1f}%')
        print(f'   Value per Dollar: ${intern_total_value/intern_costs:.2f}')
    
    total_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)
    
    print()
    print('⚡ PERFORMANCE METRICS:')
    print(f'   Total Runtime: {total_time:.2f} seconds')
    print(f'   Avg Step Time: {avg_step_time:.3f} seconds')
    
    print()
    print('🎉 WAGE ADJUSTMENT ANALYSIS COMPLETE!')
    
    return {
        'total_production': total_production,
        'total_costs': total_costs,
        'daily_profit': daily_profit,
        'labor_cost_percentage': labor_cost_percentage,
        'cost_increase': cost_increase,
        'profit_change': profit_change
    }

if __name__ == "__main__":
    results = test_wage_adjustments()
