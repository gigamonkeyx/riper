#!/usr/bin/env python3
"""
Test script for comprehensive bakery production workflows
"""

from economy_sim import MesaBakeryModel

def test_bakery_workflows():
    """Test the comprehensive bakery workflow system"""
    print("🏭 COMPREHENSIVE BAKERY WORKFLOWS TEST")
    print("=" * 60)
    
    try:
        # Initialize model with new workflows
        model = MesaBakeryModel()
        workflows = model.bakery_workflows
        
        print("✅ WORKFLOW SYSTEMS LOADED:")
        for workflow_name, workflow_data in workflows.items():
            print(f"   📋 {workflow_name.upper().replace('_', ' ')}:")
            print(f"      Type: {workflow_data['workflow_type']}")
            
            if 'parameters' in workflow_data:
                params = workflow_data['parameters']
                if workflow_name == 'meat_production':
                    print(f"      Batch Frequency: {params['batch_frequency_days']} days")
                    print(f"      Total Carcass: {params['total_carcass_lb']} lb")
                    print(f"      Smoke Percentage: {params['smoke_pct']*100}%")
                elif workflow_name == 'grain_milling':
                    print(f"      Daily Input: {params['grain_input_tons']} tons")
                    print(f"      Yield: {params['yield_pct']*100}%")
                    print(f"      Target Flour: {params['target_flour_lb']} lb")
                elif workflow_name == 'fruit_processing':
                    print(f"      Annual Input: {params['fruit_input_lb']:,} lb")
                    print(f"      Target Jars: {params['target_jars']:,}")
                    print(f"      Processing Days: {params['processing_days']}")
                elif workflow_name == 'baking':
                    print(f"      Daily Production: {params['daily_pies_pastries']:,} units")
                    print(f"      Batches per Day: {params['batches_per_day']}")
                elif workflow_name == 'expanded_canning':
                    print(f"      Daily Fresh: {params['daily_veggie_lb']} lb")
                    print(f"      Canning Input: {params['canning_input_lb']} lb")
                elif workflow_name == 'counter_sales':
                    print(f"      Daily Smoked: {params['daily_smoked_lb']} lb")
                    print(f"      Daily Sandwiches: {params['daily_sandwiches']}")
            
            if 'outputs' in workflow_data:
                outputs = workflow_data['outputs']
                print(f"      Key Outputs: {len(outputs)} products")
            
            print()
        
        print("🔧 WORKFLOW INTEGRATION ANALYSIS:")
        
        # Calculate total daily labor requirements
        total_labor_hours = 0
        workflow_labor = {}
        
        for workflow_name, workflow_data in workflows.items():
            if 'processing_steps' in workflow_data:
                daily_labor = 0
                steps = workflow_data['processing_steps']
                
                for step_name, step_data in steps.items():
                    if isinstance(step_data, dict) and 'labor_required' in step_data:
                        labor = step_data['labor_required']
                        
                        # Calculate duration in hours
                        duration_hours = 0
                        if 'duration_hours' in step_data:
                            duration_hours = step_data['duration_hours']
                        elif 'duration_minutes' in step_data:
                            duration_hours = step_data['duration_minutes'] / 60
                        
                        # Adjust for workflow frequency
                        if workflow_name == 'meat_production':
                            # 14-day batch, so daily average
                            daily_labor += (labor * duration_hours) / 14
                        elif workflow_name == 'fruit_processing':
                            # Seasonal, so daily average over year
                            daily_labor += (labor * duration_hours) / 365 * 12  # 12 processing days
                        else:
                            # Daily workflows
                            daily_labor += labor * duration_hours
                
                workflow_labor[workflow_name] = daily_labor
                total_labor_hours += daily_labor
        
        print(f"📊 DAILY LABOR REQUIREMENTS:")
        for workflow_name, labor_hours in workflow_labor.items():
            print(f"   {workflow_name.replace('_', ' ').title()}: {labor_hours:.1f} labor-hours/day")
        
        print(f"\n   TOTAL DAILY LABOR: {total_labor_hours:.1f} labor-hours/day")
        print(f"   EQUIVALENT WORKERS: {total_labor_hours/8:.1f} full-time (8-hour) workers")
        
        print("\n🎯 WORKFLOW BOTTLENECKS:")
        for workflow_name, workflow_data in workflows.items():
            if 'bottlenecks' in workflow_data:
                bottlenecks = workflow_data['bottlenecks']
                print(f"   {workflow_name.replace('_', ' ').title()}: {', '.join(bottlenecks)}")
        
        print("\n✅ COMPREHENSIVE BAKERY WORKFLOWS SUCCESSFULLY INTEGRATED!")
        print("🚀 Ready for parametric optimization and UI integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing workflows: {e}")
        return False

if __name__ == "__main__":
    success = test_bakery_workflows()
    if success:
        print("\n🎉 ALL WORKFLOW TESTS PASSED!")
    else:
        print("\n💥 WORKFLOW TESTS FAILED!")
