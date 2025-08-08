import React, { useState, useEffect, useMemo } from 'react';
import { Bar, Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_API_BASE
  ? process.env.NEXT_PUBLIC_API_BASE
  : 'http://localhost:8000/api';

const TonasketSimUI = () => {
  const [sliderDefs, setSliderDefs] = useState(null);
  const [values, setValues] = useState(null);
  const [results, setResults] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load parameter definitions and initial values from API
  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`${API_BASE}/simulation/parameters`);
        if (!res.ok) throw new Error(`Parameters HTTP ${res.status}`);
        const data = await res.json();
        setSliderDefs(data.slider_definitions);
        setValues({
          fruitCapacity: data.current_values?.fruit_capacity ?? 15000,
          jarsOutput: data.current_values?.jars_output ?? 300,
          bundlesOutput: data.current_values?.bundles_output ?? 300,
          meatProcessing: data.current_values?.meat_processing ?? 200,
          loafProduction: data.current_values?.loaf_production ?? 1166,
          wholesalePrice: data.current_values?.wholesale_price ?? 3.0,
          retailPrice: data.current_values?.retail_price ?? 5.0,
        });
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  // Calculate via API whenever values change (debounced)
  useEffect(() => {
    if (!values) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        setError(null);
        const res = await fetch(`${API_BASE}/simulation/calculate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            fruitCapacity: values.fruitCapacity,
            jarsOutput: values.jarsOutput,
            bundlesOutput: values.bundlesOutput,
            meatProcessing: values.meatProcessing,
            loafProduction: values.loafProduction,
            wholesalePrice: values.wholesalePrice,
            retailPrice: values.retailPrice,
          }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Calculate HTTP ${res.status}`);
        const data = await res.json();
        setResults(data.results);
        setBreakdown(data.breakdown);
        setTrends(data.trends);
      } catch (e) {
        if (e.name !== 'AbortError') setError(e.message);
      }
    }, 200);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [values]);

  const onChange = (key) => (e) => {
    const v = parseFloat(e.target.value);
    setValues((prev) => ({ ...prev, [key]: isNaN(v) ? prev[key] : v }));
  };

  const revenueChartData = useMemo(() => {
    const jars = (breakdown?.mason_jars ?? breakdown?.masonJars) ?? (values ? values.jarsOutput * 3 : 0);
    const bundles = breakdown?.premium_bundles ?? (values ? values.bundlesOutput * 25 : 0);
    const retailBread = breakdown?.retail_bread ?? 150;
    const wholesaleBread = breakdown?.wholesale_bread ?? 1659;
    const empanadas = breakdown?.empanadas ?? 1000;
    const customPans = breakdown?.custom_pans ?? 2000;
    return {
      labels: ['Retail Bread', 'Wholesale Bread', 'Mason Jars', 'Premium Bundles', 'Empanadas', 'Custom Pans'],
      datasets: [{
        label: 'Daily Revenue ($)',
        data: [retailBread, wholesaleBread, jars, bundles, empanadas, customPans],
        backgroundColor: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'],
        borderRadius: 8,
      }],
    };
  }, [breakdown, values]);

  const profitChartData = useMemo(() => {
    const monthly = trends?.monthlyProfits ?? [120000, 135000, 142000, 148000, 155000, (results?.dailyProfit ?? 0) * 30];
    return {
      labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
      datasets: [{
        label: 'Monthly Profit ($)',
        data: monthly,
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
      }],
    };
  }, [trends, results]);

  const complianceChartData = useMemo(() => ({
    labels: ['Meals Served', 'Free Output Value', 'Compliance Rate'],
    datasets: [{
      data: [
        (results?.mealsServed ?? 0) / 1000,
        750,
        results?.grantCompliance ?? 100,
      ],
      backgroundColor: ['#10B981', '#F59E0B', '#3B82F6'],
    }],
  }), [results]);

  if (loading) {
    return (
      <div className="min-h-screen grid place-items-center bg-gray-50">
        <div className="text-gray-700">Loading simulation UI…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen grid place-items-center bg-gray-50 p-6">
        <div className="max-w-xl w-full bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-red-600 mb-2">Error</h2>
          <p className="text-gray-700 mb-4">{error}</p>
          <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={() => location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">🍞 Tonasket Bakery Simulation</h1>
          <p className="text-lg text-gray-600">Interactive simulation with real-time parameter adjustment</p>
        </div>

        {/* Sliders Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">📊 Simulation Parameters</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Fruit Capacity */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🍎 Fruit Capacity: {values.fruitCapacity.toLocaleString()} lbs/year
              </label>
              <input type="range" min="5000" max="30000" step="1000" value={values.fruitCapacity} onChange={onChange('fruitCapacity')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>5,000</span><span>30,000</span></div>
            </div>

            {/* Mason Jars */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🫙 Mason Jars: {values.jarsOutput} jars/day
              </label>
              <input type="range" min="50" max="500" step="25" value={values.jarsOutput} onChange={onChange('jarsOutput')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>50</span><span>500</span></div>
            </div>

            {/* Premium Bundles */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                📦 Premium Bundles: {values.bundlesOutput} bundles/day
              </label>
              <input type="range" min="50" max="500" step="25" value={values.bundlesOutput} onChange={onChange('bundlesOutput')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>50</span><span>500</span></div>
            </div>

            {/* Meat Processing */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🥩 Meat Processing: {values.meatProcessing} lbs/week
              </label>
              <input type="range" min="100" max="300" step="25" value={values.meatProcessing} onChange={onChange('meatProcessing')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>100</span><span>300</span></div>
            </div>

            {/* Loaf Production */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🍞 Loaf Production: {values.loafProduction} loaves/day
              </label>
              <input type="range" min="500" max="1500" step="50" value={values.loafProduction} onChange={onChange('loafProduction')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>500</span><span>1,500</span></div>
            </div>

            {/* Wholesale Price */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                💰 Wholesale Price: ${values.wholesalePrice.toFixed(2)}/loaf
              </label>
              <input type="range" min="2.00" max="4.00" step="0.25" value={values.wholesalePrice} onChange={onChange('wholesalePrice')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>$2.00</span><span>$4.00</span></div>
            </div>

            {/* Retail Price */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🏪 Retail Price: ${values.retailPrice.toFixed(2)}/loaf
              </label>
              <input type="range" min="4.00" max="6.00" step="0.25" value={values.retailPrice} onChange={onChange('retailPrice')} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider" />
              <div className="flex justify-between text-xs text-gray-500"><span>$4.00</span><span>$6.00</span></div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Performance Summary Table */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📈 Performance Summary</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 font-medium text-gray-700">Metric</th>
                    <th className="text-right py-2 font-medium text-gray-700">Value</th>
                    <th className="text-right py-2 font-medium text-gray-700">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  <tr>
                    <td className="py-2 text-gray-600">Daily Revenue</td>
                    <td className="py-2 text-right font-medium">${(results?.dailyRevenue ?? 0).toLocaleString()}</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Daily Profit</td>
                    <td className="py-2 text-right font-medium">${(results?.dailyProfit ?? 0).toLocaleString()}</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Meals Served/Year</td>
                    <td className="py-2 text-right font-medium">{(results?.mealsServed ?? 0).toLocaleString()}</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Grant Compliance</td>
                    <td className="py-2 text-right font-medium">{results?.grantCompliance ?? 100}%</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Profit Margin</td>
                    <td className="py-2 text-right font-medium">{results?.profitMargin ?? 0}%</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Key Metrics Cards */}
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-6 text-white">
              <h4 className="text-lg font-semibold mb-2">💰 Daily Revenue</h4>
              <p className="text-3xl font-bold">${(results?.dailyRevenue ?? 0).toLocaleString()}</p>
              <p className="text-green-100 text-sm">Annual: ${(((results?.dailyRevenue ?? 0) * 365)).toLocaleString()}</p>
            </div>

            <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
              <h4 className="text-lg font-semibold mb-2">📊 Daily Profit</h4>
              <p className="text-3xl font-bold">${(results?.dailyProfit ?? 0).toLocaleString()}</p>
              <p className="text-blue-100 text-sm">Margin: {results?.profitMargin ?? 0}%</p>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-6 text-white">
              <h4 className="text-lg font-semibold mb-2">🤝 Community Impact</h4>
              <p className="text-3xl font-bold">{(((results?.mealsServed ?? 0) / 1000)).toFixed(0)}K</p>
              <p className="text-purple-100 text-sm">Meals served annually</p>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Revenue Breakdown Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📊 Revenue Breakdown</h3>
            <Bar data={revenueChartData} options={{ responsive: true, plugins: { legend: { display: false }, title: { display: false } }, scales: { y: { beginAtZero: true } } }} />
          </div>

          {/* Profit Trends Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📈 Profit Trends</h3>
            <Line data={profitChartData} options={{ responsive: true, plugins: { legend: { display: false }, title: { display: false } }, scales: { y: { beginAtZero: true } } }} />
          </div>

          {/* Compliance Overview Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">🎯 Grant Compliance</h3>
            <Pie data={complianceChartData} options={{ responsive: true, plugins: { legend: { position: 'bottom', labels: { fontSize: 12 } }, title: { display: false } } }} />
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>🍞 Tonasket Bakery Simulation - RIPER-Ω Protocol v2.6</p>
          <p>Set NEXT_PUBLIC_API_BASE to point at your API (default http://localhost:8000/api)</p>
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
      `}</style>
    </div>
  );
};

export default TonasketSimUI;
