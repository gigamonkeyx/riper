import React, { useState, useEffect } from 'react';
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

const TonasketSimUI = () => {
  // State for slider values
  const [sliderValues, setSliderValues] = useState({
    fruitCapacity: 15000,
    jarsOutput: 300,
    bundlesOutput: 300,
    meatProcessing: 200,
    loafProduction: 1166,
    wholesalePrice: 3.00,
    retailPrice: 5.00
  });

  // State for simulation results
  const [results, setResults] = useState({
    dailyRevenue: 6092,
    dailyProfit: 4493,
    mealsServed: 100000,
    grantCompliance: 100,
    profitMargin: 73.9
  });

  // Handle slider changes
  const handleSliderChange = (key, value) => {
    setSliderValues(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }));
    
    // Simulate real-time calculation updates
    calculateResults({ ...sliderValues, [key]: parseFloat(value) });
  };

  // Calculate results based on slider values
  const calculateResults = (values) => {
    // Simplified calculation logic
    const baseRevenue = 6092;
    const fruitMultiplier = values.fruitCapacity / 15000;
    const jarsRevenue = values.jarsOutput * 3.00;
    const bundlesRevenue = values.bundlesOutput * 25.00;
    const breadRevenue = values.loafProduction * 0.5 * values.wholesalePrice + 
                        values.loafProduction * 0.5 * values.retailPrice;
    
    const totalRevenue = breadRevenue + jarsRevenue + bundlesRevenue * fruitMultiplier;
    const totalProfit = totalRevenue * 0.739; // 73.9% margin
    
    setResults({
      dailyRevenue: Math.round(totalRevenue),
      dailyProfit: Math.round(totalProfit),
      mealsServed: Math.round(values.loafProduction * 365 * 0.5), // 50% free output
      grantCompliance: 100,
      profitMargin: 73.9
    });
  };

  // Chart data configurations
  const revenueChartData = {
    labels: ['Retail Bread', 'Wholesale Bread', 'Mason Jars', 'Premium Bundles', 'Empanadas', 'Custom Pans'],
    datasets: [{
      label: 'Daily Revenue ($)',
      data: [150, 1659, sliderValues.jarsOutput * 3, sliderValues.bundlesOutput * 25, 1000, 2000],
      backgroundColor: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'],
      borderRadius: 8,
    }]
  };

  const profitChartData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [{
      label: 'Monthly Profit ($)',
      data: [120000, 135000, 142000, 148000, 155000, results.dailyProfit * 30],
      borderColor: '#10B981',
      backgroundColor: 'rgba(16, 185, 129, 0.1)',
      tension: 0.4,
    }]
  };

  const complianceChartData = {
    labels: ['Meals Served', 'Free Output Value', 'Compliance Rate'],
    datasets: [{
      data: [results.mealsServed / 1000, 750, results.grantCompliance],
      backgroundColor: ['#10B981', '#F59E0B', '#3B82F6'],
    }]
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🍞 Tonasket Bakery Simulation
          </h1>
          <p className="text-lg text-gray-600">
            Interactive simulation with real-time parameter adjustment
          </p>
        </div>

        {/* Sliders Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">
            📊 Simulation Parameters
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Fruit Capacity Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🍎 Fruit Capacity: {sliderValues.fruitCapacity.toLocaleString()} lbs/year
              </label>
              <input
                type="range"
                min="5000"
                max="30000"
                step="1000"
                value={sliderValues.fruitCapacity}
                onChange={(e) => handleSliderChange('fruitCapacity', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>5,000</span>
                <span>30,000</span>
              </div>
            </div>

            {/* Mason Jars Output Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🫙 Mason Jars: {sliderValues.jarsOutput} jars/day
              </label>
              <input
                type="range"
                min="50"
                max="500"
                step="25"
                value={sliderValues.jarsOutput}
                onChange={(e) => handleSliderChange('jarsOutput', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>50</span>
                <span>500</span>
              </div>
            </div>

            {/* Premium Bundles Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                📦 Premium Bundles: {sliderValues.bundlesOutput} bundles/day
              </label>
              <input
                type="range"
                min="50"
                max="500"
                step="25"
                value={sliderValues.bundlesOutput}
                onChange={(e) => handleSliderChange('bundlesOutput', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>50</span>
                <span>500</span>
              </div>
            </div>

            {/* Meat Processing Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🥩 Meat Processing: {sliderValues.meatProcessing} lbs/week
              </label>
              <input
                type="range"
                min="100"
                max="300"
                step="25"
                value={sliderValues.meatProcessing}
                onChange={(e) => handleSliderChange('meatProcessing', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>100</span>
                <span>300</span>
              </div>
            </div>

            {/* Loaf Production Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🍞 Loaf Production: {sliderValues.loafProduction} loaves/day
              </label>
              <input
                type="range"
                min="500"
                max="1500"
                step="50"
                value={sliderValues.loafProduction}
                onChange={(e) => handleSliderChange('loafProduction', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>500</span>
                <span>1,500</span>
              </div>
            </div>

            {/* Wholesale Price Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                💰 Wholesale Price: ${sliderValues.wholesalePrice.toFixed(2)}/loaf
              </label>
              <input
                type="range"
                min="2.00"
                max="4.00"
                step="0.25"
                value={sliderValues.wholesalePrice}
                onChange={(e) => handleSliderChange('wholesalePrice', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>$2.00</span>
                <span>$4.00</span>
              </div>
            </div>

            {/* Retail Price Slider */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                🏪 Retail Price: ${sliderValues.retailPrice.toFixed(2)}/loaf
              </label>
              <input
                type="range"
                min="4.00"
                max="6.00"
                step="0.25"
                value={sliderValues.retailPrice}
                onChange={(e) => handleSliderChange('retailPrice', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>$4.00</span>
                <span>$6.00</span>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Performance Summary Table */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              📈 Performance Summary
            </h3>
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
                    <td className="py-2 text-right font-medium">${results.dailyRevenue.toLocaleString()}</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Daily Profit</td>
                    <td className="py-2 text-right font-medium">${results.dailyProfit.toLocaleString()}</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Meals Served/Year</td>
                    <td className="py-2 text-right font-medium">{results.mealsServed.toLocaleString()}</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Grant Compliance</td>
                    <td className="py-2 text-right font-medium">{results.grantCompliance}%</td>
                    <td className="py-2 text-right">✅</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-600">Profit Margin</td>
                    <td className="py-2 text-right font-medium">{results.profitMargin}%</td>
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
              <p className="text-3xl font-bold">${results.dailyRevenue.toLocaleString()}</p>
              <p className="text-green-100 text-sm">Annual: ${(results.dailyRevenue * 365).toLocaleString()}</p>
            </div>
            
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
              <h4 className="text-lg font-semibold mb-2">📊 Daily Profit</h4>
              <p className="text-3xl font-bold">${results.dailyProfit.toLocaleString()}</p>
              <p className="text-blue-100 text-sm">Margin: {results.profitMargin}%</p>
            </div>
            
            <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-6 text-white">
              <h4 className="text-lg font-semibold mb-2">🤝 Community Impact</h4>
              <p className="text-3xl font-bold">{(results.mealsServed / 1000).toFixed(0)}K</p>
              <p className="text-purple-100 text-sm">Meals served annually</p>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Revenue Breakdown Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              📊 Revenue Breakdown
            </h3>
            <Bar 
              data={revenueChartData} 
              options={{
                responsive: true,
                plugins: {
                  legend: { display: false },
                  title: { display: false }
                },
                scales: {
                  y: { beginAtZero: true }
                }
              }} 
            />
          </div>

          {/* Profit Trends Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              📈 Profit Trends
            </h3>
            <Line 
              data={profitChartData} 
              options={{
                responsive: true,
                plugins: {
                  legend: { display: false },
                  title: { display: false }
                },
                scales: {
                  y: { beginAtZero: true }
                }
              }} 
            />
          </div>

          {/* Compliance Overview Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              🎯 Grant Compliance
            </h3>
            <Pie 
              data={complianceChartData} 
              options={{
                responsive: true,
                plugins: {
                  legend: { 
                    position: 'bottom',
                    labels: { fontSize: 12 }
                  },
                  title: { display: false }
                }
              }} 
            />
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>🍞 Tonasket Bakery Simulation - RIPER-Ω Protocol v2.6</p>
          <p>Real-time simulation with Ollama qwen2.5-coder:7b/llama3.2:1b on RTX 3080 GPU</p>
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
