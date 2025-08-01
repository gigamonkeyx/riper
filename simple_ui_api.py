#!/usr/bin/env python3
"""
Simple UI API for Tonasket Bakery Simulation
Provides REST endpoints for the React UI to interact with the simulation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Tonasket Bakery Simulation API is running",
        "version": "1.0.0"
    })

@app.route('/api/simulation/parameters', methods=['GET'])
def get_parameters():
    """Get current simulation parameters"""
    return jsonify({
        "slider_definitions": {
            "fruit_capacity": {
                "min": 5000, "max": 30000, "default": 15000, "step": 1000,
                "unit": "lbs/year", "label": "Fruit Capacity"
            },
            "jars_output": {
                "min": 50, "max": 500, "default": 300, "step": 25,
                "unit": "jars/day", "label": "Mason Jars Output"
            },
            "bundles_output": {
                "min": 50, "max": 500, "default": 300, "step": 25,
                "unit": "bundles/day", "label": "Premium Bundles Output"
            },
            "meat_processing": {
                "min": 100, "max": 300, "default": 200, "step": 25,
                "unit": "lbs/week", "label": "Meat Processing"
            },
            "loaf_production": {
                "min": 500, "max": 1500, "default": 1166, "step": 50,
                "unit": "loaves/day", "label": "Loaf Production"
            },
            "wholesale_price": {
                "min": 2.00, "max": 4.00, "default": 3.00, "step": 0.25,
                "unit": "$/loaf", "label": "Wholesale Price"
            },
            "retail_price": {
                "min": 4.00, "max": 6.00, "default": 5.00, "step": 0.25,
                "unit": "$/loaf", "label": "Retail Price"
            }
        },
        "current_values": {
            "fruit_capacity": 15000, "jars_output": 300, "bundles_output": 300,
            "meat_processing": 200, "loaf_production": 1166,
            "wholesale_price": 3.00, "retail_price": 5.00
        }
    })

@app.route('/api/simulation/calculate', methods=['POST'])
def calculate_results():
    """Calculate simulation results based on input parameters"""
    try:
        data = request.get_json()
        
        # Extract parameters with defaults
        fruit_capacity = data.get('fruitCapacity', 15000)
        jars_output = data.get('jarsOutput', 300)
        bundles_output = data.get('bundlesOutput', 300)
        meat_processing = data.get('meatProcessing', 200)
        loaf_production = data.get('loafProduction', 1166)
        wholesale_price = data.get('wholesalePrice', 3.00)
        retail_price = data.get('retailPrice', 5.00)
        
        # Calculate revenue components
        retail_bread_revenue = min(30, loaf_production * 0.05) * retail_price  # 30 loaves retail cap
        wholesale_bread_revenue = (loaf_production - min(30, loaf_production * 0.05)) * wholesale_price
        jars_revenue = jars_output * 3.00  # $3.00 per jar
        bundles_revenue = bundles_output * 25.00  # $25.00 per bundle
        empanadas_revenue = (meat_processing / 7) * 2.00 * 7  # Daily empanadas from weekly meat
        flour_revenue = 1600  # Fixed flour revenue
        other_revenue = 500  # Other products
        
        # Total daily revenue
        total_revenue = (retail_bread_revenue + wholesale_bread_revenue + 
                        jars_revenue + bundles_revenue + empanadas_revenue + 
                        flour_revenue + other_revenue)
        
        # Calculate costs and profit (simplified)
        base_costs = 2600  # Base daily costs
        fruit_multiplier = fruit_capacity / 15000  # Scale costs with fruit capacity
        total_costs = base_costs * fruit_multiplier
        daily_profit = total_revenue - total_costs
        profit_margin = (daily_profit / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Calculate meals served (50% free output)
        free_loaves = loaf_production * 0.5
        free_flour_lbs = 750  # Fixed free flour
        meals_served_annually = (free_loaves + free_flour_lbs / 2) * 365
        
        # Grant compliance
        grant_compliance = 100 if meals_served_annually >= 100000 else 95
        
        # Revenue breakdown for charts
        revenue_breakdown = {
            "retail_bread": retail_bread_revenue,
            "wholesale_bread": wholesale_bread_revenue,
            "mason_jars": jars_revenue,
            "premium_bundles": bundles_revenue,
            "empanadas": empanadas_revenue,
            "flour_products": flour_revenue,
            "other": other_revenue
        }
        
        # Monthly profit trends (simulated growth)
        monthly_profits = []
        base_monthly = daily_profit * 30
        for month in range(6):
            seasonal_factor = 1.0 + (month * 0.05)  # Growth trend
            monthly_profits.append(int(base_monthly * seasonal_factor))
        
        return jsonify({
            "results": {
                "dailyRevenue": int(total_revenue),
                "dailyProfit": int(daily_profit),
                "mealsServed": int(meals_served_annually),
                "grantCompliance": grant_compliance,
                "profitMargin": round(profit_margin, 1)
            },
            "breakdown": revenue_breakdown,
            "trends": {
                "monthlyProfits": monthly_profits
            },
            "parameters": {
                "fruitCapacity": fruit_capacity,
                "jarsOutput": jars_output,
                "bundlesOutput": bundles_output,
                "meatProcessing": meat_processing,
                "loafProduction": loaf_production,
                "wholesalePrice": wholesale_price,
                "retailPrice": retail_price
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get current simulation status and metrics"""
    return jsonify({
        "status": "operational",
        "systems": {
            "ui_sliders": "active",
            "output_display": "active",
            "fruit_capacity": "active",
            "meat_locker": "active",
            "reporting": "active"
        },
        "metrics": {
            "annual_revenue": 2220000,
            "annual_profit": 1640000,
            "building_cost": 518240,
            "meals_served": 100000,
            "grant_compliance": 1.0
        },
        "fitness_score": 2.9
    })

@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to default parameters"""
    return jsonify({
        "status": "success",
        "message": "Simulation reset to default parameters"
    })

if __name__ == '__main__':
    print("🚀 Starting Tonasket Bakery Simulation API...")
    print("🌐 Starting Flask server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
