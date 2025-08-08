import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const MapNoSSR = dynamic(() => import('../sections/MapSection'), { ssr: false });

const API_BASE = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_API_BASE
  ? process.env.NEXT_PUBLIC_API_BASE
  : 'http://localhost:8000/api';

export default function Home() {
  const [params, setParams] = useState(null);
  const [values, setValues] = useState(null);
  const [results, setResults] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const res = await fetch(`${API_BASE}/simulation/parameters`);
      const data = await res.json();
      setParams(data.slider_definitions);
      setValues({
        fruitCapacity: data.current_values?.fruit_capacity ?? 15000,
        jarsOutput: data.current_values?.jars_output ?? 300,
        bundlesOutput: data.current_values?.bundles_output ?? 300,
        meatProcessing: data.current_values?.meat_processing ?? 200,
        loafProduction: data.current_values?.loaf_production ?? 1166,
        wholesalePrice: data.current_values?.wholesale_price ?? 3.0,
        retailPrice: data.current_values?.retail_price ?? 5.0,
      });
      setLoading(false);
    };
    load();
  }, []);

  useEffect(() => {
    if (!values) return;
    const timer = setTimeout(async () => {
      const res = await fetch(`${API_BASE}/simulation/calculate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values)
      });
      const data = await res.json();
      setResults(data.results);
      setBreakdown(data.breakdown);
    }, 200);
    return () => clearTimeout(timer);
  }, [values]);

  const revenueData = useMemo(() => ({
    labels: ['Retail Bread', 'Wholesale', 'Jars', 'Bundles', 'Empanadas', 'Flour'],
    datasets: [{
      label: 'Daily Revenue ($)',
      data: [
        breakdown?.retail_bread ?? 0,
        breakdown?.wholesale_bread ?? 0,
        breakdown?.mason_jars ?? 0,
        breakdown?.premium_bundles ?? 0,
        breakdown?.empanadas ?? 0,
        breakdown?.flour_products ?? 0,
      ],
      backgroundColor: ['#b23a2a', '#203a43', '#e9a23b', '#6b7e3b', '#2d5966', '#8b5e34'],
      borderRadius: 8,
    }]
  }), [breakdown]);

  return (
    <>
      <Head>
        <title>Tonasket Simulation Dashboard</title>
      </Head>
      <main className="p-6 max-w-[1400px] mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-4xl font-headline" style={{color:'var(--crate-red)'}}>TONASKET FOODS</h1>
            <p className="text-sm" style={{color:'var(--crate-navy)'}}>Rural systems • ABM • DES • SD</p>
          </div>
          <a className="crate-btn" href="/" onClick={(e)=>{e.preventDefault(); location.reload();}}>Reset</a>
        </div>

        {/* Controls + KPIs */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="crate-panel p-5 lg:col-span-2">
            <h2 className="font-headline text-xl mb-4" style={{color:'var(--crate-olive)'}}>Simulation Controls</h2>
            {loading || !params || !values ? (
              <div className="text-gray-600">Loading controls…</div>
            ) : (
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {[
                  ['fruitCapacity','Fruit Capacity (lbs/yr)',5000,30000,1000],
                  ['jarsOutput','Mason Jars (per day)',50,500,25],
                  ['bundlesOutput','Premium Bundles (per day)',50,500,25],
                  ['meatProcessing','Meat Processing (lbs/wk)',100,300,25],
                  ['loafProduction','Loaf Production (per day)',500,1500,50],
                  ['wholesalePrice','Wholesale $/loaf',2.00,4.00,0.25],
                  ['retailPrice','Retail $/loaf',4.00,6.00,0.25],
                ].map(([key,label,min,max,step]) => (
                  <div key={key} className="">
                    <div className="flex justify-between text-xs mb-1"><span className="text-gray-700">{label}</span><span className="text-gray-500">{values[key]}</span></div>
                    <input type="range" min={min} max={max} step={step} value={values[key]} onChange={(e)=>setValues(v=>({...v,[key]:parseFloat(e.target.value)}))} className="w-full"/>
                  </div>
                ))}
                <div className="col-span-full flex gap-3 pt-2">
                  <button className="crate-btn" onClick={()=>setValues(v=>({...v}))}>Recalculate</button>
                  <a className="crate-btn" style={{background:'var(--crate-olive)'}} href="#maps">Focus Maps</a>
                </div>
              </div>
            )}
          </div>
          <div className="grid gap-4">
            <div className="crate-kpi">
              <div className="text-xs opacity-80">Daily Revenue</div>
              <div className="text-3xl font-semibold">${(results?.dailyRevenue ?? 0).toLocaleString()}</div>
              <div className="text-xs opacity-70">Annual ${(results?.dailyRevenue ? results.dailyRevenue*365 : 0).toLocaleString()}</div>
            </div>
            <div className="crate-kpi crate-kpi--amber">
              <div className="text-xs opacity-90">Daily Profit</div>
              <div className="text-3xl font-semibold">${(results?.dailyProfit ?? 0).toLocaleString()}</div>
              <div className="text-xs opacity-80">Margin {results?.profitMargin ?? 0}%</div>
            </div>
            <div className="crate-kpi crate-kpi--olive">
              <div className="text-xs opacity-90">Meals Served (annual)</div>
              <div className="text-3xl font-semibold">{(results?.mealsServed ?? 0).toLocaleString()}</div>
              <div className="text-xs opacity-80">Compliance {results?.grantCompliance ?? 100}%</div>
            </div>
          </div>
        </div>

        {/* Revenue Chart */}
        <div className="crate-card p-5 mb-6">
          <h2 className="font-headline text-xl mb-4" style={{color:'var(--crate-navy)'}}>Revenue Breakdown</h2>
          <Bar data={revenueData} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} />
        </div>

        {/* Maps */}
        <div id="maps" className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="crate-card p-5">
            <h3 className="font-headline text-lg mb-3" style={{color:'var(--crate-red)'}}>Wholesale Customers Routes</h3>
            <MapNoSSR kind="customers" />
          </div>
          <div className="crate-card p-5">
            <h3 className="font-headline text-lg mb-3" style={{color:'var(--crate-red)'}}>Supplier Routes</h3>
            <MapNoSSR kind="suppliers" />
          </div>
        </div>
      </main>
    </>
  );
}
