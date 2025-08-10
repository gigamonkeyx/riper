#!/usr/bin/env python3
"""
UI API for Tonasket Bakery Simulation
Provides REST endpoints for the React UI to interact with the simulation
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import asyncio
# Attempt heavy imports; fallback to lightweight stubs if unavailable
try:
    from economy_sim import MesaBakeryModel  # type: ignore
    from economy_rewards import SDSystem  # type: ignore
    from orchestration import SimPyDESLogistics  # type: ignore
    _HEAVY_OK = True
except Exception as _imp_err:  # noqa: N816
    print(f"⚠️ Falling back to lightweight stubs for UI API: {_imp_err}")
    _HEAVY_OK = False

    class MesaBakeryModel:  # lightweight stub
        def __init__(self):
            self.ui_slider_system = {
                "slider_definitions": {
                    "fruit_capacity": {"min": 5000, "max": 30000, "default": 15000, "step": 1000, "unit": "lbs/year", "label": "Fruit Capacity"},
                    "jars_output": {"min": 50, "max": 500, "default": 300, "step": 25, "unit": "jars/day", "label": "Mason Jars Output"},
                    "bundles_output": {"min": 50, "max": 500, "default": 300, "step": 25, "unit": "bundles/day", "label": "Premium Bundles Output"},
                    "meat_processing": {"min": 100, "max": 300, "default": 200, "step": 25, "unit": "lbs/week", "label": "Meat Processing"},
                    "loaf_production": {"min": 500, "max": 1500, "default": 1166, "step": 50, "unit": "loaves/day", "label": "Loaf Production"},
                    "wholesale_price": {"min": 2.0, "max": 4.0, "default": 3.0, "step": 0.25, "unit": "$/loaf", "label": "Wholesale Price"},
                    "retail_price": {"min": 4.0, "max": 6.0, "default": 5.0, "step": 0.25, "unit": "$/loaf", "label": "Retail Price"},
                },
                "current_values": {
                    "fruit_capacity": 15000,
                    "jars_output": 300,
                    "bundles_output": 300,
                    "meat_processing": 200,
                    "loaf_production": 1166,
                    "wholesale_price": 3.0,
                    "retail_price": 5.0,
                },
                "ui_framework": "shadcn",
            }
            self.ui_calculations_system = {
                "annual_revenue": 2_220_000,
                "annual_profit": 1_640_000,
            }
            self.building_cost_system = {"total_building_cost": 518_240}

    class SDSystem:  # lightweight stub
        def __init__(self):
            # minimal reporting system for endpoints that read it
            self.reporting_system = {
                "grant_compliance_metrics": {
                    "total_meals_equivalent": 100000,
                    "compliance_rate": 1.0,
                },
                "annual_financials": {"grants_donations": 0},
                "donations_breakdown": {},
            }

        async def simulate_funding_costs(self, active_program_ids=None, override_current_year=None):
            yr = int(override_current_year or 1)
            base_hours = 120 + (yr - 1) * 30
            development = 40 + (yr - 1) * 10
            audit_extra = 10 * yr
            total = base_hours + development + audit_extra
            monthly_labor_cost = total * 45
            reimbursement_carry_cost = 500 * yr
            return {
                "programs_count": len(active_program_ids or []),
                "year": yr,
                "hours": {"base": base_hours, "development": development, "audit_extra": audit_extra, "total": total},
                "monthly_labor_cost": monthly_labor_cost,
                "reimbursement_carry_cost": reimbursement_carry_cost,
                "monthly_total_cost": monthly_labor_cost + reimbursement_carry_cost,
            }

        async def simulate_snap_wic_discount_curve(self, base_units_per_month=8000, avg_unit_price=5.0, unit_cogs=1.81, elasticity=-0.6):
            curve = []
            for d in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
                units = int(base_units_per_month * (1 + elasticity * d))
                gp = (avg_unit_price * (1 - d) - unit_cogs) * units
                curve.append({"discount": d, "units": units, "gross_profit": int(gp)})
            return {"curve": curve}

        async def simulate_donations_summary(self, base_cash_annual=50000):
            annual = {
                "net_cash": int(base_cash_annual * 0.9),
                "in_kind_total": 15000,
                "processing_fees": int(base_cash_annual * 0.03),
                "total_support": int(base_cash_annual * 0.9) + 15000,
            }
            monthly = {k: int(v/12) for k,v in annual.items()}
            return {"annual": annual, "monthly": monthly}

    class SimPyDESLogistics:  # stub
        def __init__(self):
            self.ok = True

from economy_sim import MesaBakeryModel
from economy_rewards import SDSystem
from orchestration import SimPyDESLogistics
import yaml
from pathlib import Path
import time
import logging
import os
from collections import deque
from datetime import datetime


app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load programs registry (cached in memory)
REGISTRY_PATH = Path('.riper/programs_registry.yaml')
programs_registry = {}
if REGISTRY_PATH.exists():
    try:
        with open(REGISTRY_PATH, 'r') as f:
            programs_registry = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Failed to load programs registry: {e}")
else:
    print("⚠️ Programs registry file not found; funding endpoints will be limited")
# In-memory program overrides and ranking weights
program_overrides = {}  # e.g., {"cfpcgp": {"enabled": False}}
current_weights = {
    "usefulness": {"alignment": 0.45, "eligible_uses": 0.25, "award_size": 0.20, "time_to_cash": 0.10},
    "cost": {"compliance_labor": 0.40, "reimbursement_lag": 0.25, "match_exposure": 0.20, "audit_nepa": 0.15},
}


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
        print("Simulation components initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing simulation: {e}")
        return False

# In-memory ring buffer for UI logs
LOG_BUFFER = deque(maxlen=1000)

def log_ui(message: str, level: str = "INFO"):
    ts = datetime.utcnow().isoformat() + "Z"
    entry = {"ts": ts, "level": level, "msg": message}
    LOG_BUFFER.append(entry)
    print(f"[{level}] {ts} {message}")

@app.route('/api/terminal/logs', methods=['GET'])
def get_logs():
    since = request.args.get('since')  # iso timestamp
    items = list(LOG_BUFFER)
    if since:
        items = [x for x in items if x['ts'] > since]
    return jsonify({"logs": items})

@app.route('/api/terminal/run', methods=['POST'])
def run_command():
    """Run a safe, whitelisted backend command and return output + streamed logs."""
    payload = request.get_json(silent=True) or {}
    cmd = (payload.get('cmd') or '').strip().lower()
    args = payload.get('args') or {}
    try:
        if cmd == 'status':
            log_ui("Terminal status check")
            return jsonify({"ok": True, "status": "ready"})
        elif cmd == 'recalc-costs':
            year = int(args.get('year', 1))
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            res = loop.run_until_complete(sd_system.simulate_funding_costs(override_current_year=year))
            loop.close(); log_ui(f"Recalculated costs for year {year}")
            return jsonify({"ok": True, "result": res})
        elif cmd == 'snapwic':
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            res = loop.run_until_complete(sd_system.simulate_snap_wic_discount_curve())
            loop.close(); log_ui("Computed SNAP/WIC curve")
            return jsonify({"ok": True, "result": res})
        else:
            return jsonify({"ok": False, "error": "Unknown command"}), 400
    except Exception as e:
        log_ui(f"Terminal error: {e}", level="ERROR")
        return jsonify({"ok": False, "error": str(e)}), 500
@app.route('/api/terminal/stream', methods=['GET'])
def stream_logs():
    """Server-sent events stream for UI terminal to avoid polling pauses."""
    def event_stream():
        last_index = 0
        while True:
            items = list(LOG_BUFFER)
            if last_index < len(items):
                for entry in items[last_index:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                last_index = len(items)
            time.sleep(1)
    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return Response(event_stream(), headers=headers)

@app.route('/api/terminal/heartbeat', methods=['GET'])
def heartbeat():
    """Lightweight heartbeat for terminals or editors."""
    log_ui("heartbeat")
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()+"Z"})

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
        return jsonify({"error": str(e)}), 500

# ---- Metrics helpers & endpoints -------------------------------------------------

def _compute_daily_from_current():
    """Compute daily revenue/costs/etc from current slider values."""
    ui = model.ui_slider_system
    vals = ui["current_values"]
    fruit_capacity = vals.get('fruit_capacity', 15000)
    jars_output = vals.get('jars_output', 300)
    bundles_output = vals.get('bundles_output', 300)
    meat_processing = vals.get('meat_processing', 200)
    loaf_production = vals.get('loaf_production', 1166)
    wholesale_price = vals.get('wholesale_price', 3.00)
    retail_price = vals.get('retail_price', 5.00)

    retail_bread_revenue = (loaf_production * 0.05) * retail_price
    wholesale_bread_revenue = (loaf_production * 0.95) * wholesale_price
    jars_revenue = jars_output * 3.00
    bundles_revenue = bundles_output * 25.00
    empanadas_revenue = (meat_processing / 7) * 2.00 * 7
    flour_revenue = 1600
    other_revenue = 500

    total_revenue = (
        retail_bread_revenue + wholesale_bread_revenue + jars_revenue +
        bundles_revenue + empanadas_revenue + flour_revenue + other_revenue
    )

    base_costs = 2600
    fruit_multiplier = fruit_capacity / 15000
    total_costs = base_costs * fruit_multiplier

    # Meals served estimate
    free_loaves = loaf_production * 0.5
    free_flour_lbs = 750
    meals_served_annually = (free_loaves + free_flour_lbs / 2) * 365

    return {
        "daily_revenue": float(total_revenue),
        "daily_costs": float(total_costs),
        "loaf_production": float(loaf_production),
        "meals_served_monthly": int(meals_served_annually / 12),
    }


@app.route('/api/metrics/overview', methods=['GET'])
def metrics_overview():
    """Return KPI metrics for dashboard tiles.
    - food_bank: meals served (monthly)
    - mill: monthly loaves (approx)
    - op_cost_monthly: operating costs per month
    - op_revenue_monthly: operating revenue per month
    """
    if not model:
        return jsonify({"error": "Simulation not initialized"}), 500
    try:
        daily = _compute_daily_from_current()
        op_rev_mo = int(daily["daily_revenue"] * 30)
        op_cost_mo = int(daily["daily_costs"] * 30)
        mill_monthly = int(daily["loaf_production"] * 30)
        return jsonify({
            "food_bank": daily["meals_served_monthly"],
            "mill": mill_monthly,
            "op_cost_monthly": op_cost_mo,
            "op_revenue_monthly": op_rev_mo,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics/realtime', methods=['GET'])
def metrics_realtime():
    """Return a single realtime sample for a named signal.
    Query: ?signal=grain_intake|bread_baked|donations|energy_kwh|water_gal|staffing_count
    """
    if not model:
        return jsonify({"error": "Simulation not initialized"}), 500
    try:
        import random
        signal = (request.args.get('signal') or 'bread_baked').lower()
        daily = _compute_daily_from_current()
        now = datetime.utcnow().isoformat()+"Z"
        val = 0.0
        if signal == 'bread_baked':
            base = daily["loaf_production"]
            val = base * random.uniform(0.85, 1.15)
        elif signal == 'grain_intake':
            # fruit_capacity per year -> per day -> convert lbs to kg
            ui = model.ui_slider_system
            fruit_capacity = ui["current_values"].get('fruit_capacity', 15000)
            per_day_lbs = fruit_capacity / 365.0
            per_day_kg = per_day_lbs * 0.453592
            val = per_day_kg * random.uniform(0.7, 1.3)
        elif signal == 'donations':
            # Use reporting system if available, else random base
            base = 800.0
            try:
                base = float(sd_system.reporting_system.get('donations_breakdown', {}).get('daily_cash', 800.0))
            except Exception:
                pass
            val = base * random.uniform(0.6, 1.4)
        elif signal == 'energy_kwh':
            # Rough energy baseline proportional to production and facility size
            base = daily["loaf_production"] * 0.8 + 250.0
            val = base * random.uniform(0.7, 1.3)
        elif signal == 'water_gal':
            base = daily["loaf_production"] * 0.3 + 120.0
            val = base * random.uniform(0.7, 1.3)
        elif signal == 'staffing_count':
            base = 8.0
            val = base * random.uniform(0.8, 1.2)
        else:
            val = random.uniform(10, 100)
        return jsonify({"signal": signal, "value": int(val), "ts": now})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        return jsonify({"error": str(e)}), 500

@app.route('/api/funding/programs', methods=['GET'])
def list_programs():
    """Return programs from central registry (with optional filters)"""
    if not programs_registry:
        return jsonify({"programs": []})
    raw = programs_registry.get('programs', {})
    # Optional filter: enabled=true/false
    enabled = request.args.get('enabled')
    items = []
    # Apply in-memory overrides to /programs too
    effective = {}
    for pid, v in raw.items():
        merged = dict(v)
        if pid in program_overrides:
            merged.update(program_overrides[pid])
        effective[pid] = merged

    for key, val in effective.items():
        if enabled is not None:
            flag = str(val.get('enabled', True)).lower()
            if flag != str(enabled).lower():
                continue
        items.append({"id": key, **val})
    return jsonify({"programs": items})

@app.route('/api/funding/rank', methods=['GET'])
def rank_programs():
    """Rank programs by usefulness and cost-to-service, with configurable weights"""
    if not programs_registry:
        return jsonify({"programs": [], "weights": {}})

    raw = programs_registry.get('programs', {})
    # Apply in-memory overrides (e.g., enabled flags)
    effective = {}
    for pid, val in raw.items():
        merged = dict(val)
        if pid in program_overrides:
            merged.update(program_overrides[pid])
        effective[pid] = merged

    # Default weights (approved by user)
    weights = current_weights

    # Simple scoring heuristics based on registry fields; real values will be refined
    def score_program(pid: str, p: dict):
        # Usefulness
        alignment = 1.0 if any(tag in ["underserved","food_security","access","facility","value_added","processing","organic"] for tag in p.get('mission_tags', [])) else 0.6
        eligible_uses = 1.0 if p.get('type') in ("grant","loan_grant_mix","in_kind_plus_admin") else 0.7
        award_span = p.get('typical_award_max') or p.get('grant_share_up_to', 0)
        award_size = min(1.0, (award_span or 0) / 2000000)  # normalized to $2M cap
        time_to_cash = 0.5 if p.get('reimbursement', True) else 0.9  # advances are faster

        usefulness = (
            alignment * weights['usefulness']['alignment'] +
            eligible_uses * weights['usefulness']['eligible_uses'] +
            award_size * weights['usefulness']['award_size'] +
            time_to_cash * weights['usefulness']['time_to_cash']
        )

        # Cost-to-service (lower is better, invert to create a score to sort separately)
        compliance_labor = 0.8 if 'quarterly' in (p.get('reporting_cadence') or []) else 0.5
        reimbursement_lag = 0.8 if p.get('reimbursement', True) else 0.3
        match_exposure = 0.8 if (p.get('cost_share_rule','').strip() not in ("", "No fixed match; leverage encouraged")) else 0.3
        audit_nepa = 0.8 if p.get('audit_sensitivity') in ("high",) or 'NEPA' in (p.get('procurement_rigor') or '') else 0.4

        cost = (
            compliance_labor * weights['cost']['compliance_labor'] +
            reimbursement_lag * weights['cost']['reimbursement_lag'] +
            match_exposure * weights['cost']['match_exposure'] +
            audit_nepa * weights['cost']['audit_nepa']
        )

        return usefulness, cost

    items = []
    for pid, p in effective.items():
        if not p.get('enabled', True):
            pass  # still include but marked disabled
        u, c = score_program(pid, p)
        items.append({"id": pid, **p, "usefulness_score": round(u, 3), "cost_to_service_score": round(c, 3)})

    # Sort primarily by usefulness (desc), secondarily by cost (asc)
    items_sorted = sorted(items, key=lambda x: (-x['usefulness_score'], x['cost_to_service_score']))
    return jsonify({"programs": items_sorted, "weights": weights})
@app.route('/api/funding/costs', methods=['GET'])
def funding_costs():
    """Estimate monthly funding service costs using SDSystem.simulate_funding_costs"""
    if not sd_system:
        return jsonify({"error": "Simulation not initialized"}), 500
    try:
        programs = request.args.get('programs')
        year = request.args.get('year')
        active = programs.split(',') if programs else [
            'cfpcgp','lfpp','vapg','omdg','rbdg','community_facilities'
        ]
        # SDSystem.simulate_funding_costs is async; run in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(sd_system.simulate_funding_costs(active_program_ids=active, override_current_year=int(year) if year else None))
        loop.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/funding/snapwic-benefit', methods=['GET'])
def snapwic_benefit():
    """Return cost-benefit curve for SNAP/WIC discounts on staple bread/COGS"""
    if not sd_system:
        return jsonify({"error": "Simulation not initialized"}), 500
    try:
        base_units = float(request.args.get('base_units', 8000))
        avg_price = float(request.args.get('avg_price', 5.00))
        unit_cogs = float(request.args.get('unit_cogs', 1.81))
        elasticity = float(request.args.get('elasticity', -0.6))
        # SDSystem.simulate_snap_wic_discount_curve is async
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            sd_system.simulate_snap_wic_discount_curve(
                base_units_per_month=int(base_units),
                avg_unit_price=avg_price,
                unit_cogs=unit_cogs,
                elasticity=elasticity
            )
        )
        loop.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/funding/donations-summary', methods=['GET'])
def donations_summary():
    """Return donations summary (cash, in-kind, fees)"""
    if not sd_system:
        return jsonify({"error": "Simulation not initialized"}), 500
    try:
        base_cash = float(request.args.get('base_cash_annual', 50000))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(sd_system.simulate_donations_summary(base_cash_annual=base_cash))
        loop.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/funding/apply-donations', methods=['POST'])
def apply_donations():
    """Apply donations to reporting_system annual figures (updates grants_donations and adds breakdown)."""
    if not sd_system:
        return jsonify({"error": "Simulation not initialized"}), 500
    try:
        payload = request.get_json(silent=True) or {}
        base_cash = float(payload.get('base_cash_annual', 50000))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary = loop.run_until_complete(sd_system.simulate_donations_summary(base_cash_annual=base_cash))
        loop.close()
        # Update reporting system totals
        sd_system.reporting_system.setdefault('donations_breakdown', {})
        sd_system.reporting_system['donations_breakdown'] = summary
        sd_system.reporting_system['annual_financials']['grants_donations'] = summary['annual']['total_support']
        return jsonify({
            "updated": True,
            "grants_donations": sd_system.reporting_system['annual_financials']['grants_donations'],
            "donations": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/funding/toggle', methods=['POST'])
def toggle_program():
    """In-memory enable/disable toggle for funding programs (session-only)."""
    try:
        payload = request.get_json(silent=True) or {}
        pid = payload.get('id')
        enabled = payload.get('enabled')
        if not pid or enabled is None:
            return jsonify({"error": "id and enabled required"}), 400
        program_overrides.setdefault(pid, {})['enabled'] = bool(enabled)
        return jsonify({"id": pid, "enabled": bool(enabled)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/funding/weights', methods=['GET','PUT'])
def weights_endpoint():
    """Get or set in-memory ranking weights."""
    global current_weights
    if request.method == 'GET':
        return jsonify(current_weights)
    try:
        payload = request.get_json(silent=True) or {}
        # Validate and update shallowly
        for group in ('usefulness','cost'):
            if group in payload and isinstance(payload[group], dict):
                current_weights[group].update({k: float(v) for k, v in payload[group].items()})
        return jsonify(current_weights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/funding/save-overrides', methods=['POST'])
def save_overrides():
    """Persist program enabled overrides to .riper/programs_registry.yaml (with backup)."""
    try:
        if not programs_registry:
            return jsonify({"error": "Registry not loaded"}), 500
        # Load current file, merge enabled flags, write back
        data = dict(programs_registry)
        programs = data.get('programs', {})
        for pid, override in (program_overrides or {}).items():
            if pid in programs and 'enabled' in override:
                programs[pid]['enabled'] = bool(override['enabled'])
        # Backup
        backup_path = REGISTRY_PATH.with_suffix('.backup.yaml')
        try:
            with open(backup_path, 'w') as bf:
                yaml.safe_dump(programs_registry, bf, sort_keys=False)
        except Exception as e:
            print(f"⚠️ Backup failed: {e}")
        # Write
        with open(REGISTRY_PATH, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return jsonify({"saved": True, "path": str(REGISTRY_PATH)})
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
            }
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
    print("Starting Tonasket Bakery Simulation API...")

    # Initialize simulation components
    if initialize_simulation():
        port = int(os.environ.get('PORT', '8000'))
        print(f"Starting Flask server on http://localhost:{port}")
        log_ui("server started")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("Failed to initialize simulation. Exiting.")
        exit(1)
