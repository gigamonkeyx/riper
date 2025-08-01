#!/usr/bin/env python3
"""
Labor Cost Breakdown Analysis for 500-Agent Bread-Focused Tonasket Simulation
"""

from economy_sim import MesaBakeryModel
import time

def analyze_labor_costs():
    print('💰 LABOR COST BREAKDOWN ANALYSIS')
    print('=' * 60)

    # Create 500-agent model
    model = MesaBakeryModel(
        num_bakers=25, num_participants=50, num_customers=125, 
        num_labor=125, num_suppliers=125, num_partners=125,
        num_c_corps=5, num_llcs=10, num_gov_entities=5,
        width=25, height=25
    )

    # Run a few steps to get realistic data
    for i in range(3):
        model.step()

    print(f'Total Labor Agents: {len(model.labor_agents)}')
    print('=' * 60)

    # Analyze by labor type
    bakers = [a for a in model.labor_agents if a.labor_type == "baker"]
    interns = [a for a in model.labor_agents if a.labor_type == "intern"]
    staff = [a for a in model.labor_agents if a.labor_type == "staff"]

    print('👨‍🍳 BAKERS BREAKDOWN:')
    print(f'   Count: {len(bakers)} agents')
    if bakers:
        baker_costs = [a.daily_wage_cost for a in bakers]
        baker_production = [a.bread_items_produced for a in bakers]
        baker_hours = [a.hours_per_day for a in bakers]
        baker_wages = [a.hourly_wage for a in bakers]
        
        print(f'   Total Daily Cost: ${sum(baker_costs):.2f}')
        print(f'   Average Daily Cost: ${sum(baker_costs)/len(bakers):.2f}')
        print(f'   Hourly Wage Range: ${min(baker_wages):.2f} - ${max(baker_wages):.2f}')
        print(f'   Hours per Day: {baker_hours[0]} hours (standard)')
        print(f'   Total Bread Production: {sum(baker_production)} items')
        print(f'   Avg Production per Baker: {sum(baker_production)/len(bakers):.1f} items')
        print(f'   Cost per Bread Item: ${sum(baker_costs)/max(1, sum(baker_production)):.2f}')

    print('\n🎓 INTERNS BREAKDOWN:')
    print(f'   Count: {len(interns)} agents')
    if interns:
        intern_costs = [a.daily_wage_cost for a in interns]
        intern_production = [a.bread_items_produced for a in interns]
        intern_hours = [a.hours_per_day for a in interns]
        intern_wages = [a.hourly_wage for a in interns]
        
        print(f'   Total Daily Cost: ${sum(intern_costs):.2f}')
        print(f'   Average Daily Cost: ${sum(intern_costs)/len(interns):.2f}')
        print(f'   Hourly Wage: ${intern_wages[0]:.2f} (minimum wage)')
        print(f'   Hours per Day: {intern_hours[0]} hours (part-time)')
        print(f'   Total Bread Production: {sum(intern_production)} items')
        print(f'   Avg Production per Intern: {sum(intern_production)/len(interns):.1f} items')
        print(f'   Cost per Bread Item: ${sum(intern_costs)/max(1, sum(intern_production)):.2f}')

    print('\n👥 STAFF BREAKDOWN:')
    print(f'   Count: {len(staff)} agents')
    if staff:
        staff_costs = [a.daily_wage_cost for a in staff]
        staff_production = [a.bread_items_produced for a in staff]
        staff_hours = [a.hours_per_day for a in staff]
        staff_wages = [a.hourly_wage for a in staff]
        
        print(f'   Total Daily Cost: ${sum(staff_costs):.2f}')
        print(f'   Average Daily Cost: ${sum(staff_costs)/len(staff):.2f}')
        print(f'   Hourly Wage Range: ${min(staff_wages):.2f} - ${max(staff_wages):.2f}')
        print(f'   Hours per Day: {staff_hours[0]} hours (full-time)')
        print(f'   Total Bread Production: {sum(staff_production)} items')
        print(f'   Avg Production per Staff: {sum(staff_production)/len(staff):.1f} items')
        print(f'   Cost per Bread Item: ${sum(staff_costs)/max(1, sum(staff_production)):.2f}')

    # Overall analysis
    total_costs = sum(a.daily_wage_cost for a in model.labor_agents)
    total_production = sum(a.bread_items_produced for a in model.labor_agents)
    total_hours = sum(a.hours_per_day for a in model.labor_agents)

    print('\n' + '=' * 60)
    print('📊 OVERALL LABOR ANALYSIS:')
    print(f'   Total Daily Labor Cost: ${total_costs:.2f}')
    print(f'   Total Daily Production: {total_production} bread items')
    print(f'   Total Daily Hours: {total_hours} hours')
    print(f'   Average Cost per Hour: ${total_costs/total_hours:.2f}')
    print(f'   Average Cost per Bread Item: ${total_costs/max(1, total_production):.2f}')
    print(f'   Labor Efficiency: {total_production/total_hours:.2f} items/hour')

    # Cost efficiency analysis
    print('\n💡 COST EFFICIENCY ANALYSIS:')
    if bakers and total_production > 0:
        baker_efficiency = sum(baker_production) / sum(baker_costs)
        intern_efficiency = sum(intern_production) / max(1, sum(intern_costs))
        staff_efficiency = sum(staff_production) / max(1, sum(staff_costs))
        
        print(f'   Bakers: {baker_efficiency:.2f} items per dollar')
        print(f'   Interns: {intern_efficiency:.2f} items per dollar')
        print(f'   Staff: {staff_efficiency:.2f} items per dollar')
        
        # Recommendations
        print('\n🎯 OPTIMIZATION RECOMMENDATIONS:')
        if baker_efficiency > intern_efficiency and baker_efficiency > staff_efficiency:
            print('   ✅ Bakers are most cost-efficient - consider hiring more bakers')
        elif intern_efficiency > baker_efficiency:
            print('   ✅ Interns are most cost-efficient - expand internship program')
        else:
            print('   ✅ Staff are most cost-efficient - maintain current staff levels')

    # Revenue analysis
    bread_price = 5.0  # $5 per loaf
    daily_revenue = total_production * bread_price
    labor_cost_percentage = (total_costs / daily_revenue) * 100 if daily_revenue > 0 else 0

    print('\n💰 REVENUE ANALYSIS:')
    print(f'   Daily Bread Revenue: ${daily_revenue:.2f} ({total_production} × ${bread_price})')
    print(f'   Labor Cost Percentage: {labor_cost_percentage:.1f}% of revenue')
    print(f'   Daily Profit (before other costs): ${daily_revenue - total_costs:.2f}')
    
    if labor_cost_percentage < 30:
        print('   ✅ Labor costs are well-controlled (<30% of revenue)')
    elif labor_cost_percentage < 50:
        print('   ⚠️  Labor costs are moderate (30-50% of revenue)')
    else:
        print('   🔴 Labor costs are high (>50% of revenue) - optimization needed')

    print('\n' + '=' * 60)
    print('💰 LABOR COST BREAKDOWN ANALYSIS COMPLETE')

if __name__ == "__main__":
    analyze_labor_costs()
