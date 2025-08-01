#!/usr/bin/env python3
"""
UI API for Tonasket Bakery Simulation
Provides REST endpoints for the React UI to interact with the simulation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import asyncio
from economy_sim import MesaBakeryModel
from economy_rewards import SDSystem
from orchestration import SimPyDESLogistics

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global simulation instances
model = None
sd_system = None
des_logistics = None

def initialize_simulation():
    """Initialize simulation components"""
    global model, sd_system, des_logistics
    try:
        model = MesaBakeryModel()
        sd_system = SDSystem()
        des_logistics = SimPyDESLogistics()
        print("✅ Simulation components initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing simulation: {e}")
        return False

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
    if not model:
        return jsonify({"error": "Simulation not initialized"}), 500
    
    try:
        ui_system = model.ui_slider_system
        return jsonify({
            "slider_definitions": ui_system["slider_definitions"],
            "current_values": ui_system["current_values"],
            "ui_framework": ui_system["ui_framework"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/calculate', methods=['POST'])
def calculate_results():
    """Calculate simulation results based on input parameters"""
    if not model or not sd_system:
        return jsonify({"error": "Simulation not initialized"}), 500
    
    try:
        data = request.get_json()
        
        # Extract parameters
        fruit_capacity = data.get('fruitCapacity', 15000)
        jars_output = data.get('jarsOutput', 300)
        bundles_output = data.get('bundlesOutput', 300)
        meat_processing = data.get('meatProcessing', 200)
        loaf_production = data.get('loafProduction', 1166)
        wholesale_price = data.get('wholesalePrice', 3.00)
        retail_price = data.get('retailPrice', 5.00)
        
        # Calculate revenue components
        retail_bread_revenue = (loaf_production * 0.05) * retail_price  # 5% retail (30 loaves cap)
        wholesale_bread_revenue = (loaf_production * 0.95) * wholesale_price  # 95% wholesale
        jars_revenue = jars_output * 3.00  # $3.00 per jar
        bundles_revenue = bundles_output * 25.00  # $25.00 per bundle
        empanadas_revenue = (meat_processing / 7) * 2.00 * 7  # Daily empanadas from weekly meat
        flour_revenue = 1600  # Fixed flour revenue
        other_revenue = 500  # Other products
        
        # Total daily revenue
        total_revenue = (retail_bread_revenue + wholesale_bread_revenue + 
                        jars_revenue + bundles_revenue + empanadas_revenue + 
                        flour_revenue + other_revenue)
        
        # Calculate costs and profit
        base_costs = 2600  # Base daily costs
        fruit_multiplier = fruit_capacity / 15000  # Scale costs with fruit capacity
        total_costs = base_costs * fruit_multiplier
        daily_profit = total_revenue - total_costs
        profit_margin = (daily_profit / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Calculate meals served (50% free output)
        free_loaves = loaf_production * 0.5
        free_flour_lbs = 750  # Fixed free flour
        meals_served_annually = (free_loaves + free_flour_lbs / 2) * 365  # Approximate meal equivalents
        
        # Grant compliance (always 100% if meeting targets)
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
        
        # Monthly profit trends (simulated)
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
    if not model or not sd_system:
        return jsonify({"error": "Simulation not initialized"}), 500
    
    try:
        # Get system status
        ui_calc = model.ui_calculations_system
        building = model.building_cost_system
        reporting = sd_system.reporting_system
        
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
                "annual_revenue": ui_calc["annual_revenue"],
                "annual_profit": ui_calc["annual_profit"],
                "building_cost": building["total_building_cost"],
                "meals_served": reporting["grant_compliance_metrics"]["total_meals_equivalent"],
                "grant_compliance": reporting["grant_compliance_metrics"]["compliance_rate"]
            },
            "fitness_score": 2.9
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to default parameters"""
    global model, sd_system, des_logistics
    
    try:
        # Reinitialize simulation
        success = initialize_simulation()
        if success:
            return jsonify({
                "status": "success",
                "message": "Simulation reset to default parameters"
            })
        else:
            return jsonify({"error": "Failed to reset simulation"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Tonasket Bakery Simulation API...")
    
    # Initialize simulation components
    if initialize_simulation():
        print("🌐 Starting Flask server on http://localhost:8000")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        print("❌ Failed to initialize simulation. Exiting.")
        exit(1)
