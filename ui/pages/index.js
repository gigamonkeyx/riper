import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useState } from 'react';
// Use dynamic import for charts to avoid SSR issues
const BarNoSSR = dynamic(() => import('react-chartjs-2').then(m => m.Bar), { ssr: false });
import MiniSparkline from "@/components/MiniSparkline";
import BarChart from "@/components/BarChart";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Filler, Title, Tooltip, Legend } from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Filler, Title, Tooltip, Legend);
import AppShell from '../components/AppShell';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import RealtimeMeter from "@/components/RealtimeMeter";
import { Slider } from "@/components/ui/slider";

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Filler, Title, Tooltip, Legend);

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

  // Fetch KPI overview once on load
  useEffect(() => {
    const run = async () => {
      try {
        const res = await fetch(`${API_BASE}/metrics/overview`);
        if (res.ok) {
          const m = await res.json();
          setResults((r)=>({
            ...r,
            foodBank: m.food_bank,
            mill: m.mill,
            opCostMonthly: m.op_cost_monthly,
            opRevenueMonthly: m.op_revenue_monthly,
          }));
        }
      } catch {}
    };
    run();
  }, []);

  // Share KPIs via store for other pages
  useEffect(()=>{
    try {
      import("@/lib/metricsStore").then(({ useMetricsStore })=>{
        const setKpis = useMetricsStore.getState().setKpis;
        if (results?.foodBank || results?.mill || results?.opCostMonthly || results?.opRevenueMonthly) {
          setKpis({
            food_bank: results?.foodBank ?? 0,
            mill: results?.mill ?? 0,
            op_cost_monthly: results?.opCostMonthly ?? 0,
            op_revenue_monthly: results?.opRevenueMonthly ?? 0,
          });
        }
      });
    } catch {}
  }, [results?.foodBank, results?.mill, results?.opCostMonthly, results?.opRevenueMonthly]);

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
      backgroundColor: ['#7c3aed', '#22d3ee', '#64748b', '#a78bfa', '#38bdf8', '#475569'],
      borderRadius: 8,
    }]
  }), [breakdown]);

  const profitVsRevenueData = useMemo(() => ({
    labels: ['Revenue','Profit'],
    datasets: [{
      label: 'Amount ($)',
      data: [results?.dailyRevenue ?? 0, results?.dailyProfit ?? 0],
      backgroundColor: ['#7c3aed', '#10b981'],
      borderRadius: 8,
    }]
  }), [results?.dailyRevenue, results?.dailyProfit]);

  const monthlyOpsData = useMemo(() => ({
    labels: ['Op Cost (mo)','Op Revenue (mo)'],
    datasets: [{
      label: 'Monthly ($)',
      data: [results?.opCostMonthly ?? 0, results?.opRevenueMonthly ?? 0],
      backgroundColor: ['#ef4444', '#22c55e'],
      borderRadius: 8,
    }]
  }), [results?.opCostMonthly, results?.opRevenueMonthly]);


  return (
    <>
      <Head>
        <title>Tonasket Simulation Dashboard</title>
      </Head>
      <AppShell>
        <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">TONASKET FOODS</h1>
            <p className="text-sm text-muted-foreground">Rural systems • ABM • DES • SD</p>
          </div>
          <div className="hidden md:flex gap-2">
            <a className="btn-cyber" href="/funding">Funding</a>
            <a className="btn-cyber" href="/terminal">Terminal</a>
            <a className="btn-cyber" href="/" onClick={(e)=>{e.preventDefault(); location.reload();}}>Reset</a>
          </div>
        </div>


        {/* Charts below: 2x2 grid (each ~1/4 page on desktop) */}
        <div className="max-w-[1200px] mx-auto w-full px-2">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <div className="crate-card p-5">
              <h3 className="text-lg font-medium mb-3">Profit vs Revenue</h3>
              <div className="h-64"><BarChart data={profitVsRevenueData} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} /></div>
            </div>
            <div className="crate-card p-5">
              <h3 className="text-lg font-medium mb-3">Monthly Ops</h3>
              <div className="h-64"><BarChart data={monthlyOpsData} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} /></div>
            </div>
            <div className="crate-card p-5">
              <h3 className="text-lg font-medium mb-3">Revenue Breakdown</h3>
              <div className="h-64"><BarChart data={revenueData} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} /></div>
            </div>
            <div className="crate-card p-5">
              <h3 className="text-lg font-medium mb-3">Subscriptions (Mini)</h3>
              <div className="h-64"><BarChart data={{ labels:['A','B','C','D','E','F','G'], datasets:[{ data:[3,5,8,13,21,8,10], backgroundColor:'#93c5fd' }] }} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} /></div>
            </div>
          </div>
        </div>

        {/* Top KPIs row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="rounded-lg border p-5">
            <div className="text-xs text-muted-foreground">Daily Revenue</div>
            <div className="text-3xl font-semibold">${(results?.dailyRevenue ?? 0).toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Annual {(results?.dailyRevenue ? (results.dailyRevenue*365) : 0).toLocaleString()}</div>
          </div>
          <div className="rounded-lg border p-5">
            <div className="text-xs text-muted-foreground">Daily Profit</div>
            <div className="text-3xl font-semibold">${(results?.dailyProfit ?? 0).toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Margin {results?.profitMargin ?? 0}%</div>
          </div>
          <div className="rounded-lg border p-5">
            <div className="text-xs text-muted-foreground">Meals Served (annual)</div>
            <div className="text-3xl font-semibold">{(results?.mealsServed ?? 0).toLocaleString()}</div>
          </div>
          <div className="rounded-lg border p-5">
            <div className="text-xs text-muted-foreground">Compliance</div>
            <div className="text-3xl font-semibold">{results?.grantCompliance ?? 100}%</div>
          </div>
        </div>


        {/* Simulator Studio */}
        <Tabs defaultValue="controls">
          <div className="mb-4">
            <TabsList>
              <TabsTrigger value="controls">Controls</TabsTrigger>
              <TabsTrigger value="kpis">KPIs</TabsTrigger>
              <TabsTrigger value="revenue">Revenue</TabsTrigger>
              <TabsTrigger value="costs">Costs</TabsTrigger>
              <TabsTrigger value="compliance">Compliance</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="controls">
            <ResizablePanelGroup direction="horizontal" className="min-h-[420px]">
              <ResizablePanel defaultSize={36} minSize={26} className="pr-4">
                <Card>
                  <CardContent className="p-4 space-y-3">
                    <h2 className="text-xl font-semibold mb-2">Simulation Controls</h2>
                    {loading || !params || !values ? (
                      <div className="text-muted-foreground">Loading controls…</div>
                    ) : (
                      <div className="grid sm:grid-cols-2 gap-3">
                        {[
                          ['fruitCapacity','Fruit Capacity (lbs/yr)',5000,30000,1000],
                          ['jarsOutput','Mason Jars (per day)',50,500,25],
                          ['bundlesOutput','Premium Bundles (per day)',50,500,25],
                          ['meatProcessing','Meat Processing (lbs/wk)',100,300,25],
                          ['loafProduction','Loaf Production (per day)',500,1500,50],
                          ['wholesalePrice','Wholesale $/loaf',2.00,4.00,0.25],
                          ['retailPrice','Retail $/loaf',4.00,6.00,0.25],
                        ].map(([key,label,min,max,step]) => (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between text-xs text-muted-foreground"><span>{label}</span><span className="text-foreground">{values[key]}</span></div>
                            <Slider min={min} max={max} step={step} value={[values[key]]} onValueChange={(arr)=>setValues(v=>({...v,[key]:parseFloat(arr[0])}))} />
                          </div>
                        ))}
                        <div className="col-span-full flex gap-3 pt-2">
                          <Button onClick={()=>setValues(v=>({...v}))}>Recalculate</Button>
                          <Button variant="secondary" asChild>
                            <a href="#maps">Focus Maps</a>
                          </Button>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </ResizablePanel>
              <ResizableHandle withHandle />
              <ResizablePanel defaultSize={64} minSize={38} className="pl-4">
                <Card>
                  <CardContent className="p-4">
                    <h2 className="text-xl font-semibold mb-2">KPIs</h2>
                    <div className="grid sm:grid-cols-3 gap-3">
                      <div className="rounded-lg border p-4">
                        <div className="text-xs text-muted-foreground">Daily Revenue</div>
                        <div className="text-3xl font-semibold">${(results?.dailyRevenue ?? 0).toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">Annual {(results?.dailyRevenue ? results.dailyRevenue*365 : 0).toLocaleString()}</div>
                      </div>
                      <div className="rounded-lg border p-4">
                        <div className="text-xs text-muted-foreground">Daily Profit</div>
                        <div className="text-3xl font-semibold">${(results?.dailyProfit ?? 0).toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">Margin {results?.profitMargin ?? 0}%</div>
                      </div>
                      <div className="rounded-lg border p-4">
                        <div className="text-xs text-muted-foreground">Meals Served (annual)</div>
                        <div className="text-3xl font-semibold">{(results?.mealsServed ?? 0).toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">Compliance {results?.grantCompliance ?? 100}%</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </ResizablePanel>
            </ResizablePanelGroup>
          </TabsContent>

          <TabsContent value="kpis">
            <div className="grid sm:grid-cols-3 gap-3">
              {/* Duplicate KPI grid for dedicated tab */}
            </div>
          </TabsContent>

          <TabsContent value="revenue">
            {/* Existing Revenue chart section remains below; we keep this tab for future split */}
          </TabsContent>

          <TabsContent value="costs">
            {/* Costs placeholder for now */}
          </TabsContent>

          <TabsContent value="compliance">
            {/* Compliance placeholder for now */}
          </TabsContent>
        </Tabs>

        {/* Charts and summaries below (old section removed to meet new layout) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="crate-card p-5">
            <h3 className="text-lg font-medium mb-3">Total Visitors</h3>
            <div className="h-64"><BarChart data={revenueData} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} /></div>
          </div>
          <Card>
            <CardContent className="p-5">
              <div className="text-sm text-muted-foreground mb-1">Subscriptions</div>
              <div className="text-2xl font-semibold">+{(results?.subscriptions ?? 2350).toLocaleString()}</div>
              <div className="mt-3">
                <div className="h-24"><BarChart data={{ labels: ['A','B','C','D','E','F','G'], datasets:[{ data: [3,5,8,13,21,8,10], backgroundColor:['#93c5fd','#a78bfa'] }] }} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{display:false},x:{display:false}}}} /></div>
              </div>
            </CardContent>
          </Card>
        </div>
        <Card>
          <CardContent className="p-5">
            <div className="text-sm text-muted-foreground mb-1">Alerts</div>
            <ul className="text-sm space-y-2">
              <li className="flex justify-between"><span>Compliance notice</span><span className="text-muted-foreground">New</span></li>
              <li className="flex justify-between"><span>Donations spike</span><span className="text-muted-foreground">1h</span></li>
              <li className="flex justify-between"><span>Wholesale discount</span><span className="text-muted-foreground">Today</span></li>
            </ul>
          </CardContent>
        </Card>

        {/* Additional meters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <RealtimeMeter title="Energy" unit="kWh" color="#f59e0b" signal="energy_kwh" />
          <RealtimeMeter title="Water" unit="gal" color="#60a5fa" signal="water_gal" />
          <RealtimeMeter title="Staffing" unit="people" color="#34d399" signal="staffing_count" />
        </div>
        <div className="crate-card p-5 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-accent-foreground">Revenue Breakdown</h2>
          <div className="h-64"><BarChart data={revenueData} options={{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}} /></div>
        </div>

        {/* Maps */}
        <div id="maps" className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* 24h Real-time meters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <RealtimeMeter title="Grain Intake" unit="kg" color="#7c3aed" signal="grain_intake" />
          <RealtimeMeter title="Bread Baked" unit="loaves" color="#22d3ee" signal="bread_baked" />
          <RealtimeMeter title="Donations" unit="$" color="#10b981" signal="donations" />
        </div>
          <div className="crate-card p-5">
            <h3 className="text-lg font-medium mb-3 text-muted-foreground">Wholesale Customers Routes</h3>
            <MapNoSSR kind="customers" />
          </div>
          <div className="crate-card p-5">
            <h3 className="text-lg font-medium mb-3 text-muted-foreground">Supplier Routes</h3>
            <MapNoSSR kind="suppliers" />
          </div>
        </div>
        </div>
        {/* close inner container */}
      </AppShell>
    </>
  );
}
