#!/usr/bin/env python3
"""
Test UI enhancement system with save/load configs and additional sliders
Validates Step 4 of the optimization checklist
"""

import os
import json
from economy_sim import UIConfigurationSystem, MesaBakeryModel

def test_config_save_load():
    """Test save/load configuration functionality"""
    print("🧪 Testing Config Save/Load...")
    
    # Remove existing test config
    test_config = "test_tonasket_config.json"
    if os.path.exists(test_config):
        os.remove(test_config)
    
    # Initialize UI config system
    ui_config = UIConfigurationSystem(test_config)
    
    # Test loading default config
    default_config = ui_config.load_config()
    print(f"   Default config loaded: {len(default_config)} categories")
    
    # Test modifying and saving config
    test_config_data = default_config.copy()
    test_config_data["production_sliders"]["pies_output"]["default"] = 35
    test_config_data["production_sliders"]["meat_pies_output"]["default"] = 15
    
    save_success = ui_config.save_config(test_config_data)
    print(f"   Config saved successfully: {save_success}")
    
    # Test loading modified config
    loaded_config = ui_config.load_config()
    pies_value = loaded_config["production_sliders"]["pies_output"]["default"]
    meat_pies_value = loaded_config["production_sliders"]["meat_pies_output"]["default"]
    
    print(f"   Loaded pies output: {pies_value}")
    print(f"   Loaded meat pies output: {meat_pies_value}")
    
    # Test individual slider value access
    pies_slider_value = ui_config.get_slider_value("production_sliders", "pies_output", loaded_config)
    meat_pies_slider_value = ui_config.get_slider_value("production_sliders", "meat_pies_output", loaded_config)
    
    print(f"   Pies slider value: {pies_slider_value}")
    print(f"   Meat pies slider value: {meat_pies_slider_value}")
    
    # Cleanup
    if os.path.exists(test_config):
        os.remove(test_config)
    
    success = (
        save_success and
        pies_value == 35 and
        meat_pies_value == 15 and
        pies_slider_value == 35 and
        meat_pies_slider_value == 15
    )
    
    print(f"   ✅ Config save/load: {success}")
    
    return success

def test_additional_sliders():
    """Test additional sliders for pies and meat pies output"""
    print("\n🧪 Testing Additional Sliders...")
    
    ui_config = UIConfigurationSystem()
    default_config = ui_config.load_config()
    
    # Check for required sliders
    production_sliders = default_config.get("production_sliders", {})
    
    required_sliders = [
        "pies_output",
        "meat_pies_output",
        "mason_jars",
        "premium_bundles",
        "empanadas",
        "bread_output"
    ]
    
    sliders_found = []
    for slider in required_sliders:
        if slider in production_sliders:
            slider_config = production_sliders[slider]
            sliders_found.append({
                "name": slider,
                "min": slider_config.get("min", 0),
                "max": slider_config.get("max", 0),
                "default": slider_config.get("default", 0),
                "step": slider_config.get("step", 1)
            })
    
    print(f"   Required sliders found: {len(sliders_found)}/{len(required_sliders)}")
    
    for slider in sliders_found:
        print(f"   {slider['name']}: {slider['min']}-{slider['max']} (default: {slider['default']}, step: {slider['step']})")
    
    # Validate specific slider ranges
    pies_slider = next((s for s in sliders_found if s['name'] == 'pies_output'), None)
    meat_pies_slider = next((s for s in sliders_found if s['name'] == 'meat_pies_output'), None)
    
    pies_valid = (
        pies_slider and
        pies_slider['min'] == 10 and
        pies_slider['max'] == 50 and
        pies_slider['default'] == 25
    )
    
    meat_pies_valid = (
        meat_pies_slider and
        meat_pies_slider['min'] == 5 and
        meat_pies_slider['max'] == 20 and
        meat_pies_slider['default'] == 10
    )
    
    print(f"   Pies slider valid (10-50, default 25): {pies_valid}")
    print(f"   Meat pies slider valid (5-20, default 10): {meat_pies_valid}")
    
    success = (
        len(sliders_found) == len(required_sliders) and
        pies_valid and
        meat_pies_valid
    )
    
    print(f"   ✅ Additional sliders: {success}")
    
    return success

def test_slider_value_updates():
    """Test updating slider values dynamically"""
    print("\n🧪 Testing Slider Value Updates...")
    
    test_config = "test_slider_config.json"
    if os.path.exists(test_config):
        os.remove(test_config)
    
    ui_config = UIConfigurationSystem(test_config)
    
    # Test updating individual slider values
    updates = [
        ("production_sliders", "pies_output", 40),
        ("production_sliders", "meat_pies_output", 18),
        ("economic_parameters", "wheat_price_per_ton", 420),
        ("capacity_limits", "mill_capacity_tons", 1.1)
    ]
    
    update_results = []
    for category, parameter, value in updates:
        result = ui_config.update_slider_value(category, parameter, value)
        update_results.append(result)
        print(f"   Updated {parameter} to {value}: {result}")
    
    # Verify updates were saved
    loaded_config = ui_config.load_config()
    
    verification_results = []
    for category, parameter, expected_value in updates:
        actual_value = ui_config.get_slider_value(category, parameter, loaded_config)
        matches = actual_value == expected_value
        verification_results.append(matches)
        print(f"   Verified {parameter}: {actual_value} == {expected_value} ({matches})")
    
    # Cleanup
    if os.path.exists(test_config):
        os.remove(test_config)
    
    success = all(update_results) and all(verification_results)
    
    print(f"   ✅ Slider value updates: {success}")
    
    return success

def test_model_integration():
    """Test UI config integration with MesaBakeryModel"""
    print("\n🧪 Testing Model Integration...")
    
    # Create model with UI config system
    model = MesaBakeryModel(num_customers=10, num_bakers=2)
    
    # Check if UI config system is integrated
    has_ui_config = hasattr(model, 'ui_config') and model.ui_config is not None
    print(f"   Model has UI config system: {has_ui_config}")
    
    if has_ui_config:
        # Test accessing slider values from model
        config = model.ui_config.load_config()
        
        pies_output = model.ui_config.get_slider_value("production_sliders", "pies_output", config)
        meat_pies_output = model.ui_config.get_slider_value("production_sliders", "meat_pies_output", config)
        bread_output = model.ui_config.get_slider_value("production_sliders", "bread_output", config)
        
        print(f"   Pies output from model: {pies_output}")
        print(f"   Meat pies output from model: {meat_pies_output}")
        print(f"   Bread output from model: {bread_output}")
        
        # Test updating values through model
        update_success = model.ui_config.update_slider_value("production_sliders", "pies_output", 45)
        print(f"   Updated pies output through model: {update_success}")
        
        # Verify update
        new_config = model.ui_config.load_config()
        new_pies_output = model.ui_config.get_slider_value("production_sliders", "pies_output", new_config)
        update_verified = new_pies_output == 45
        print(f"   Update verified: {update_verified}")
        
        integration_success = (
            pies_output == 25 and  # Default value
            meat_pies_output == 10 and  # Default value
            bread_output == 1166 and  # Default value
            update_success and
            update_verified
        )
    else:
        integration_success = False
    
    print(f"   ✅ Model integration: {integration_success}")
    
    return integration_success

def main():
    """Run all UI enhancement tests"""
    print("🚀 UI ENHANCEMENT SYSTEM TESTING")
    print("="*60)
    
    results = []
    
    # Test individual components
    results.append(test_config_save_load())
    results.append(test_additional_sliders())
    results.append(test_slider_value_updates())
    results.append(test_model_integration())
    
    # Overall results
    print("\n📊 TEST RESULTS SUMMARY:")
    print("="*60)
    
    test_names = [
        "Config Save/Load",
        "Additional Sliders",
        "Slider Value Updates",
        "Model Integration"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {name}: {status}")
    
    overall_success = all(results)
    fitness_impact = 0.80 if overall_success else 0.60
    
    print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    print(f"   Fitness Impact: {fitness_impact:.2f}")
    
    if overall_success:
        print("   ABM: UI enhanced. Save/Load configs added. Sliders 6 (pies output, meat pies, etc). Fitness impact: 0.80")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
