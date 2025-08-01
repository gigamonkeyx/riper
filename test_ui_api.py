#!/usr/bin/env python3
"""
Test script for the UI API
"""

import requests
import json

def test_ui_api():
    """Test the UI API endpoints"""
    base_url = "http://localhost:8000/api"
    
    print("🧪 TESTING UI API")
    print("=" * 40)
    
    try:
        # Test health endpoint
        print("1. Testing Health Endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
            data = response.json()
            print(f"   Status: {data['status']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test parameters endpoint
        print("\n2. Testing Parameters Endpoint...")
        response = requests.get(f"{base_url}/simulation/parameters")
        if response.status_code == 200:
            print("   ✅ Parameters endpoint working")
            data = response.json()
            print(f"   Sliders available: {len(data['slider_definitions'])}")
        else:
            print(f"   ❌ Parameters failed: {response.status_code}")
            return False
        
        # Test calculation endpoint
        print("\n3. Testing Calculation Endpoint...")
        test_data = {
            'fruitCapacity': 20000,
            'jarsOutput': 400,
            'bundlesOutput': 250,
            'loafProduction': 1000,
            'wholesalePrice': 3.50,
            'retailPrice': 5.50
        }
        
        response = requests.post(f"{base_url}/simulation/calculate", json=test_data)
        if response.status_code == 200:
            print("   ✅ Calculation endpoint working")
            result = response.json()
            print(f"   Daily Revenue: ${result['results']['dailyRevenue']:,}")
            print(f"   Daily Profit: ${result['results']['dailyProfit']:,}")
            print(f"   Meals Served: {result['results']['mealsServed']:,}/year")
        else:
            print(f"   ❌ Calculation failed: {response.status_code}")
            return False
        
        # Test status endpoint
        print("\n4. Testing Status Endpoint...")
        response = requests.get(f"{base_url}/simulation/status")
        if response.status_code == 200:
            print("   ✅ Status endpoint working")
            data = response.json()
            print(f"   Fitness Score: {data['fitness_score']}")
        else:
            print(f"   ❌ Status failed: {response.status_code}")
            return False
        
        print("\n🎉 ALL API TESTS PASSED!")
        print("✅ UI API is fully operational")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure simple_ui_api.py is running.")
        return False
    except Exception as e:
        print(f"❌ API Test Error: {e}")
        return False

if __name__ == "__main__":
    test_ui_api()
