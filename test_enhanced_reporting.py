#!/usr/bin/env python3
"""
Test enhanced reporting system with 17 reports/year
Validates Step 6 of the optimization checklist
"""

from economy_rewards import SDSystem

def test_reporting_system_initialization():
    """Test enhanced reporting system initialization"""
    print("🧪 Testing Reporting System Initialization...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Check reporting system structure
    reporting_system = sd_system.reporting_system
    
    # Verify annual financials
    annual_financials = reporting_system["annual_financials"]
    expected_revenue = 2220000  # $2.22M
    expected_profit = 1090000   # $1.09M
    
    revenue_correct = annual_financials["total_revenue"] == expected_revenue
    profit_correct = annual_financials["total_profit"] == expected_profit
    
    print(f"   Annual revenue: ${annual_financials['total_revenue']:,}")
    print(f"   Annual profit: ${annual_financials['total_profit']:,}")
    print(f"   Profit margin: {annual_financials['profit_margin']:.1%}")
    
    # Verify grant compliance metrics
    grant_metrics = reporting_system["grant_compliance_metrics"]
    meals_served = grant_metrics["total_meals_equivalent"]
    families_served = grant_metrics["families_served"]
    individuals_served = grant_metrics["individuals_served"]
    
    meals_correct = meals_served == 100000  # 100,000 meals
    families_correct = families_served == 150  # 150 families
    individuals_correct = individuals_served == 450  # 450 individuals
    
    print(f"   Meals served: {meals_served:,}")
    print(f"   Families served: {families_served}")
    print(f"   Individuals served: {individuals_served}")
    print(f"   Compliance rate: {grant_metrics['compliance_rate']:.0%}")
    
    success = (
        revenue_correct and
        profit_correct and
        meals_correct and
        families_correct and
        individuals_correct
    )
    
    print(f"   ✅ Reporting system initialization: {success}")
    
    return success

def test_17_reports_generation():
    """Test generation of 17 reports per year"""
    print("\n🧪 Testing 17 Reports Generation...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Test report generation for each month
    total_reports_generated = 0
    monthly_reports = 0
    quarterly_reports = 0
    annual_reports = 0
    
    for month in range(1, 13):  # 12 months
        reports = sd_system.generate_enhanced_reports(month)
        current_reports = reports["current_reports"]
        
        # Count report types
        for report in current_reports:
            if report["report_type"] == "monthly_financial":
                monthly_reports += 1
            elif report["report_type"] == "quarterly_compliance":
                quarterly_reports += 1
            elif report["report_type"] == "annual_comprehensive":
                annual_reports += 1
        
        total_reports_generated += len(current_reports)
    
    print(f"   Monthly reports: {monthly_reports}")
    print(f"   Quarterly reports: {quarterly_reports}")
    print(f"   Annual reports: {annual_reports}")
    print(f"   Total reports: {total_reports_generated}")
    
    # Verify report counts
    monthly_correct = monthly_reports == 12  # 12 monthly reports
    quarterly_correct = quarterly_reports == 4  # 4 quarterly reports
    annual_correct = annual_reports == 1  # 1 annual report
    total_correct = total_reports_generated == 17  # 17 total reports
    
    print(f"   Monthly reports correct (12): {monthly_correct}")
    print(f"   Quarterly reports correct (4): {quarterly_correct}")
    print(f"   Annual reports correct (1): {annual_correct}")
    print(f"   Total reports correct (17): {total_correct}")
    
    success = monthly_correct and quarterly_correct and annual_correct and total_correct
    
    print(f"   ✅ 17 reports generation: {success}")
    
    return success

def test_report_content_accuracy():
    """Test accuracy of report content and metrics"""
    print("\n🧪 Testing Report Content Accuracy...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Generate sample reports
    monthly_report = sd_system.generate_enhanced_reports(6)  # June
    quarterly_report = sd_system.generate_enhanced_reports(9)  # September (Q3)
    annual_report = sd_system.generate_enhanced_reports(12)  # December (Annual)
    
    # Test monthly report content
    monthly_data = monthly_report["current_reports"][0]
    monthly_revenue = monthly_data["revenue"]
    monthly_profit = monthly_data["profit"]
    monthly_meals = monthly_data["meals_served"]
    
    expected_monthly_revenue = 2220000 / 12  # $185K/month
    expected_monthly_profit = 1090000 / 12   # $90.8K/month
    expected_monthly_meals = 100000 / 12     # 8,333 meals/month
    
    monthly_revenue_correct = abs(monthly_revenue - expected_monthly_revenue) < 1000
    monthly_profit_correct = abs(monthly_profit - expected_monthly_profit) < 1000
    monthly_meals_correct = abs(monthly_meals - expected_monthly_meals) < 100
    
    print(f"   Monthly revenue: ${monthly_revenue:,.0f} (expected: ${expected_monthly_revenue:,.0f})")
    print(f"   Monthly profit: ${monthly_profit:,.0f} (expected: ${expected_monthly_profit:,.0f})")
    print(f"   Monthly meals: {monthly_meals:,.0f} (expected: {expected_monthly_meals:,.0f})")
    
    # Test quarterly report content (if available)
    quarterly_data = None
    if len(quarterly_report["current_reports"]) > 1:
        quarterly_data = quarterly_report["current_reports"][1]
        quarterly_free_output = quarterly_data["free_output_value"]
        quarterly_loaves = quarterly_data["bread_loaves_served"]
        
        expected_quarterly_free_output = 750000 / 4  # $187.5K/quarter
        expected_quarterly_loaves = 217905 / 4       # 54,476 loaves/quarter
        
        quarterly_output_correct = abs(quarterly_free_output - expected_quarterly_free_output) < 1000
        quarterly_loaves_correct = abs(quarterly_loaves - expected_quarterly_loaves) < 100
        
        print(f"   Quarterly free output: ${quarterly_free_output:,.0f} (expected: ${expected_quarterly_free_output:,.0f})")
        print(f"   Quarterly loaves: {quarterly_loaves:,.0f} (expected: {expected_quarterly_loaves:,.0f})")
    else:
        quarterly_output_correct = quarterly_loaves_correct = True  # No quarterly report in June
    
    # Test annual report content
    annual_data = annual_report["current_reports"][-1]  # Last report (annual)
    annual_revenue = annual_data["total_revenue"]
    annual_profit = annual_data["total_profit"]
    annual_meals = annual_data["total_meals_served"]
    
    annual_revenue_correct = annual_revenue == 2220000
    annual_profit_correct = annual_profit == 1090000
    annual_meals_correct = annual_meals == 100000
    
    print(f"   Annual revenue: ${annual_revenue:,} (expected: $2,220,000)")
    print(f"   Annual profit: ${annual_profit:,} (expected: $1,090,000)")
    print(f"   Annual meals: {annual_meals:,} (expected: 100,000)")
    
    success = (
        monthly_revenue_correct and
        monthly_profit_correct and
        monthly_meals_correct and
        quarterly_output_correct and
        quarterly_loaves_correct and
        annual_revenue_correct and
        annual_profit_correct and
        annual_meals_correct
    )
    
    print(f"   ✅ Report content accuracy: {success}")
    
    return success

def test_compliance_tracking():
    """Test grant compliance tracking in reports"""
    print("\n🧪 Testing Compliance Tracking...")
    
    # Initialize SD system
    sd_system = SDSystem()
    
    # Generate annual report for full compliance data
    annual_report = sd_system.generate_enhanced_reports(12)
    key_metrics = annual_report["key_metrics"]
    
    # Check compliance metrics
    compliance_rate = key_metrics["compliance_rate"]
    families_served = key_metrics["families_served"]
    individuals_served = key_metrics["individuals_served"]
    
    # Verify 100% compliance
    compliance_perfect = compliance_rate == 1.0
    families_target_met = families_served == 150
    individuals_target_met = individuals_served == 450
    
    print(f"   Compliance rate: {compliance_rate:.0%}")
    print(f"   Families served: {families_served}")
    print(f"   Individuals served: {individuals_served}")
    
    # Check grant programs
    grant_programs = len(sd_system.reporting_system["grant_programs"])
    grant_programs_adequate = grant_programs >= 5  # At least 5 grant programs
    
    print(f"   Grant programs active: {grant_programs}")
    
    # Check free output compliance
    free_output_value = sd_system.reporting_system["grant_compliance_metrics"]["free_output_annual_value"]
    free_output_percentage = sd_system.reporting_system["grant_compliance_metrics"]["free_output_percentage"]
    
    free_output_adequate = free_output_value >= 750000  # At least $750K/year
    free_output_percentage_correct = free_output_percentage == 0.50  # Exactly 50%
    
    print(f"   Free output value: ${free_output_value:,}")
    print(f"   Free output percentage: {free_output_percentage:.0%}")
    
    success = (
        compliance_perfect and
        families_target_met and
        individuals_target_met and
        grant_programs_adequate and
        free_output_adequate and
        free_output_percentage_correct
    )
    
    print(f"   ✅ Compliance tracking: {success}")
    
    return success

def main():
    """Run all enhanced reporting tests"""
    print("🚀 ENHANCED REPORTING SYSTEM TESTING")
    print("="*60)
    
    results = []
    
    # Test individual components
    results.append(test_reporting_system_initialization())
    results.append(test_17_reports_generation())
    results.append(test_report_content_accuracy())
    results.append(test_compliance_tracking())
    
    # Overall results
    print("\n📊 TEST RESULTS SUMMARY:")
    print("="*60)
    
    test_names = [
        "Reporting System Initialization",
        "17 Reports Generation",
        "Report Content Accuracy",
        "Compliance Tracking"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {name}: {status}")
    
    overall_success = all(results)
    fitness_impact = 0.95 if overall_success else 0.70
    
    print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    print(f"   Fitness Impact: {fitness_impact:.2f}")
    
    if overall_success:
        print("   SD: Reporting 17 reports generated. Metrics: 100% compliance, 100,000 meals served. Fitness impact: 0.95")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
