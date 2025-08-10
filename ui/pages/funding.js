import Head from 'next/head';
import { useEffect, useMemo, useState } from 'react';
import AppShell from '../components/AppShell';
import { Switch } from "@/components/ui/switch";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const API_BASE = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_API_BASE
  ? process.env.NEXT_PUBLIC_API_BASE
  : 'http://localhost:8000/api';

import { Line, Bar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend);

function SnapChart({ data }){
  const labels = data.curve.map(r => `${Math.round(r.discount*100)}%`);
  const gp = data.curve.map(r => r.gross_profit);
  const units = data.curve.map(r => r.units);
  const ds = {
    labels,
    datasets: [
      { label: 'Gross Profit', data: gp, borderColor: 'rgb(16, 185, 129)', backgroundColor: 'rgba(16,185,129,0.2)' },
      { label: 'Units', data: units, borderColor: 'rgb(59, 130, 246)', backgroundColor: 'rgba(59,130,246,0.2)', yAxisID: 'y1' },
    ]
  };
  const options = {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    stacked: false,
    scales: {
      y: { type: 'linear', position: 'left' },
      y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false } }
    }
  };
  return <Line data={ds} options={options} />;
}

// Top-level DonationsChart (fix scope so FundingPage can use it)
function DonationsChart({ data }){
  const labels = ['Annual'];
  const ds = {
    labels,
    datasets: [
      { label: 'Net Cash', data: [data.annual.net_cash], backgroundColor: 'rgba(34,211,238,0.6)' },
      { label: 'In-kind', data: [data.annual.in_kind_total], backgroundColor: 'rgba(168,85,247,0.6)' },
      { label: 'Fees', data: [data.annual.processing_fees], backgroundColor: 'rgba(255,45,149,0.6)' }
    ]
  };
  const options = { responsive: true, plugins: { legend: { position: 'bottom' } }, scales: { x: { }, y: { beginAtZero: true } } };
  return <Bar data={ds} options={options} />;
}


export default function FundingPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [items, setItems] = useState([]);
  const [showDisabled, setShowDisabled] = useState(true);
  const [onlyEnabled, setOnlyEnabled] = useState(false);
  const [serviceCost, setServiceCost] = useState(null);
  const [costsYear, setCostsYear] = useState(1);
  const [snapCurve, setSnapCurve] = useState(null);
  const [donations, setDonations] = useState(null);
  const [baseCash, setBaseCash] = useState(50000);
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [weights, setWeights] = useState(null);
  const [pendingWeights, setPendingWeights] = useState(null);
  const [selected, setSelected] = useState(null);
  const [baseCost, setBaseCost] = useState(null);
  const [saveMsg, setSaveMsg] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const q = onlyEnabled ? '?enabled=true' : '';
        const res = await fetch(`${API_BASE}/funding/rank${q}`);
        if (!res.ok) throw new Error(`Funding HTTP ${res.status}`);
        const data = await res.json();
        setItems(data.programs || []);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [onlyEnabled]);

  const refreshCosts = async (yr) => {
    try {
      const [resCur, resBase] = await Promise.all([
        fetch(`${API_BASE}/funding/costs?year=${yr}`),
        fetch(`${API_BASE}/funding/costs?year=1`)
      ]);
      if (!resCur.ok) throw new Error(`Costs HTTP ${resCur.status}`);
      if (!resBase.ok) throw new Error(`Costs HTTP ${resBase.status}`);
      const data = await resCur.json();
      const base = await resBase.json();
      setServiceCost(data);
      setBaseCost(base);
    } catch (e) {}
  };

  useEffect(() => {
    refreshCosts(costsYear);
  }, [costsYear]);

  useEffect(() => {
    const loadSnap = async () => {
      try {
        const res = await fetch(`${API_BASE}/funding/snapwic-benefit`);
        if (!res.ok) throw new Error(`SNAP/WIC HTTP ${res.status}`);
        const data = await res.json();
        setSnapCurve(data);
      } catch (e) {
        // ignore initial errors
      }
    };
    loadSnap();
  }, []);
  useEffect(()=>{
    (async ()=>{
      try{const res = await fetch(`${API_BASE}/funding/weights`); if(res.ok){ const w = await res.json(); setWeights(w); setPendingWeights(JSON.parse(JSON.stringify(w))); }}catch(e){}
    })();
  },[]);


  useEffect(() => {
    const loadDonations = async () => {
      try {
        const res = await fetch(`${API_BASE}/funding/donations-summary?base_cash_annual=${baseCash}`);
        if (!res.ok) throw new Error(`Donations HTTP ${res.status}`);
        const data = await res.json();
        setDonations(data);
      } catch (e) {}
    };
    loadDonations();
  }, [baseCash]);

  const visible = useMemo(() => {
    let arr = showDisabled ? items : items.filter(p => p.enabled !== false);
    if (categoryFilter !== 'all') arr = arr.filter(p => (p.category||'').toLowerCase() === categoryFilter);
    if (typeFilter !== 'all') arr = arr.filter(p => (p.type||'').toLowerCase() === typeFilter);
    return arr;
  }, [items, showDisabled, categoryFilter, typeFilter]);

  return (
    <>
      <Head>
        <title>Funding & Grants</title>
      </Head>
      <AppShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-headline" style={{color:'var(--crate-red)'}}>Funding & Grants</h1>
            <p className="text-sm text-muted-foreground">Manage funding programs and toggles. Filters and summaries below.</p>
          </div>
          <a className="btn-cyber" href="/">Back</a>
        </div>

        <Card className="mb-4">
          <CardContent className="p-5 flex flex-wrap gap-4 items-center">
            <label className="flex items-center gap-2">
              <Switch checked={onlyEnabled} onCheckedChange={(v)=>setOnlyEnabled(!!v)} />
              <span>Only show enabled programs</span>
            </label>
            <label className="flex items-center gap-2">
              <Switch checked={showDisabled} onCheckedChange={(v)=>setShowDisabled(!!v)} />
              <span>Include disabled (e.g., REAP)</span>
            </label>
            <div className="flex items-center gap-2">
              <span className="text-sm">Category:</span>
              <Select value={categoryFilter} onValueChange={(v)=>setCategoryFilter(v)}>
                <SelectTrigger className="w-44"><SelectValue placeholder="All" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="food_systems">Food Systems</SelectItem>
                  <SelectItem value="infrastructure">Infrastructure</SelectItem>
                  <SelectItem value="energy">Energy</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm">Type:</span>
              <Select value={typeFilter} onValueChange={(v)=>setTypeFilter(v)}>
                <SelectTrigger className="w-40"><SelectValue placeholder="All" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="grant">Grant</SelectItem>
                  <SelectItem value="loan_grant_mix">Loan+Grant</SelectItem>
                  <SelectItem value="authorization">Authorization</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button onClick={async()=>{await fetch(`${API_BASE}/funding/save-overrides`,{method:'POST'}); setSaveMsg('Saved'); setTimeout(()=>setSaveMsg(''), 1200);}}>Save toggles</Button>
            {saveMsg && <span className="text-green-400 text-sm">{saveMsg}</span>}
            {loading && <span className="text-[color:var(--muted-400)]">Loading…</span>}
            {error && <span className="text-red-400">Error: {error}</span>}
          </CardContent>
        </Card>

        <Card className="mb-4" data-testid="weights-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Ranking Weights</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-end mb-3">
              <Button onClick={async()=>{
                try{
                  const res = await fetch(`${API_BASE}/funding/weights`, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(pendingWeights)});
                  if(!res.ok) throw new Error('save weights failed');
                  const saved = await res.json(); setWeights(saved);
                }catch(e){}
              }}>Apply</Button>
            </div>
            {!pendingWeights && <div className="text-sm text-muted-foreground">Loading…</div>}
            {pendingWeights && (
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                {['usefulness','cost'].map(group => (
                  <div key={group}>
                    <div className="font-medium mb-1 capitalize">{group}</div>
                    {Object.keys(pendingWeights[group]||{}).map(k => (
                      <div key={k} className="mb-2">
                        <div className="flex justify-between"><span>{k}</span><span>{pendingWeights[group][k]}</span></div>
                        <input type="range" min="0" max="1" step="0.05" value={pendingWeights[group][k]} onChange={e=>setPendingWeights(w=>({...w,[group]:{...w[group],[k]:parseFloat(e.target.value)}}))} className="w-full"/>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="mb-4" data-testid="costs-card">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle>Monthly cost to service</CardTitle>
            <div className="flex items-center gap-2 text-sm">
              <span>View year:</span>
              <Select value={String(costsYear)} onValueChange={(v)=>setCostsYear(Number(v))}>
                <SelectTrigger className="w-24"><SelectValue placeholder="1" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1</SelectItem>
                  <SelectItem value="2">2</SelectItem>
                  <SelectItem value="3">3</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {!serviceCost && <div className="text-sm text-muted-foreground">Loading…</div>}
            {serviceCost && (
              <div className="text-sm grid grid-cols-2 gap-2">
                <div>Programs modeled: <b>{serviceCost.programs_count}</b></div>
                <div>Year: <b>{serviceCost.year}</b></div>
                <div>Hours (base): <b>{serviceCost.hours.base}</b></div>
                <div>Hours (development): <b>{serviceCost.hours.development}</b></div>
                <div>Hours (audit extra): <b>{serviceCost.hours.audit_extra}</b></div>
                <div>Total hours: <b>{serviceCost.hours.total}</b></div>
                <div>Labor cost: <b>${serviceCost.monthly_labor_cost.toFixed(0)}</b></div>
                <div>Reimbursement carry: <b>${serviceCost.reimbursement_carry_cost.toFixed(0)}</b></div>
                <div>Monthly total: <b>${serviceCost.monthly_total_cost.toFixed(0)}</b></div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="text-xs text-muted-foreground mb-2">
          {serviceCost && serviceCost.year === 3 && baseCost ? (
            <span title="Audit burden applied for Year 3 compared to Year 1 baseline">
              Audit burden applied. Δ hours = {serviceCost.hours.total - baseCost.hours.total}
            </span>
          ) : <span className="opacity-70">Base year view</span>}
        </div>

        <div className="crate-panel p-4 mb-4">
          <h2 className="text-lg font-medium mb-2">SNAP/WIC discount cost–benefit (bread)</h2>
          {!snapCurve && <div className="text-sm text-gray-500">Loading…</div>}
          {snapCurve && (
            <div className="text-sm overflow-x-auto">
              <div className="mb-3">
                <SnapChart data={snapCurve} />
              </div>
              <table className="min-w-[520px] text-sm">
                <thead className="text-left border-b">
                  <tr>
                    <th className="py-1 pr-4">Discount</th>
                    <th className="py-1 pr-4">Units</th>
                    <th className="py-1 pr-4">Price</th>
                    <th className="py-1 pr-4">Revenue</th>
                    <th className="py-1 pr-4">Gross Profit</th>
                  </tr>
                </thead>
                <tbody>
                  {snapCurve.curve.map((r, idx) => (
                    <tr key={idx} className="border-b">
                      <td className="py-1 pr-4">{Math.round(r.discount*100)}%</td>
                      <td className="py-1 pr-4">{r.units.toFixed(0)}</td>
                      <td className="py-1 pr-4">${r.price.toFixed(2)}</td>
                      <td className="py-1 pr-4">${r.revenue.toFixed(0)}</td>
                      <td className="py-1 pr-4">${r.gross_profit.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-2 text-xs text-gray-600">Best (by gross profit): {Math.round(snapCurve.best.discount*100)}% discount</div>
            </div>
          )}
        </div>

        <div className="crate-panel p-4 mb-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium mb-2">Donations summary</h2>
            <div className="flex items-center gap-2 text-sm">
              <span>Annual cash goal:</span>
              <input type="number" className="border px-2 py-1 rounded w-28 bg-[color:var(--panel-800)] text-[color:var(--text-100)]" value={baseCash} onChange={e=>setBaseCash(Number(e.target.value||0))} />
              <button className="btn-cyber" onClick={async()=>{
                try {
                  const res = await fetch(`${API_BASE}/funding/apply-donations`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({base_cash_annual: baseCash})});
                  if (!res.ok) throw new Error('Apply donations failed');
                  const data = await res.json();
                  setDonations(data.donations || donations);
                } catch (e) {}
              }}>Apply to reports</button>
            </div>
          </div>
          {!donations && <div className="text-sm text-gray-500">Loading…</div>}
          {donations && (
            <div>
              <div className="grid grid-cols-2 gap-2 text-sm mb-3">
                <div>Annual Gross Cash: <b>${donations.annual.gross_cash.toFixed(0)}</b></div>
                <div>Annual Realized: <b>${donations.annual.realized_cash.toFixed(0)}</b></div>
                <div>Annual Fees: <b>${donations.annual.processing_fees.toFixed(0)}</b></div>
                <div>Annual Net Cash: <b>${donations.annual.net_cash.toFixed(0)}</b></div>
                <div>Annual In-kind: <b>${donations.annual.in_kind_total.toFixed(0)}</b></div>
                <div>Total Support (Annual): <b>${donations.annual.total_support.toFixed(0)}</b></div>
              </div>
              <DonationsChart data={donations} />
            </div>
          )}
        </div>

        <div className="crate-card p-2">
          <Table>
            <TableHeader>
              <TableRow className="text-muted-foreground">
                <TableHead>Program</TableHead>
                <TableHead>Enabled</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Match Rule</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {visible.map(p => (
                <TableRow key={p.id}>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      <button className="underline" onClick={()=>setSelected(p)}>{p.name}</button>
                      {!p.enabled && <span className="text-xs px-2 py-0.5 rounded bg-muted text-foreground/80">disabled</span>}
                    </div>
                    <div className="text-xs text-muted-foreground">{(p.mission_tags||[]).join(', ')}</div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Switch checked={!!p.enabled} onCheckedChange={async (v)=>{
                        try {
                          const res = await fetch(`${API_BASE}/funding/toggle`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: p.id, enabled: v})});
                          if (!res.ok) throw new Error('toggle failed');
                          await Promise.all([
                            fetch(`${API_BASE}/funding/rank${onlyEnabled ? '?enabled=true':''}`).then(r=>r.json()).then(d=> setItems(d.programs||[])),
                            refreshCosts(costsYear)
                          ]);
                        } catch(e) {}
                      }} />
                      <span className="text-xs text-muted-foreground">{p.enabled ? 'Yes' : 'No'}</span>
                    </div>
                  </TableCell>
                  <TableCell>{p.category}</TableCell>
                  <TableCell>{p.type}</TableCell>
                  <TableCell>{p.cost_share_rule || '-'}</TableCell>
                </TableRow>
              ))}
              {!loading && visible.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">No programs to display</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        {/* Program Details Sheet via shadcn Dialog */}
        <Dialog open={!!selected} onOpenChange={(o)=>!o && setSelected(null)}>
          <DialogContent>
            {selected && (
              <div>
                <DialogHeader>
                  <DialogTitle>{selected.name}</DialogTitle>
                </DialogHeader>
                <div className="text-sm space-y-1 mt-2">
                  <div><b>Category:</b> {selected.category}</div>
                  <div><b>Type:</b> {selected.type}</div>
                  <div><b>Awards:</b> up to ${selected.typical_award_max?.toLocaleString?.()||'-'}</div>
                  <div><b>Match Rule:</b> {selected.cost_share_rule||'-'}</div>
                  <div><b>Cadence:</b> {(selected.reporting_cadence||[]).join(', ')||'-'}</div>
                  <div><b>Audit Sensitivity:</b> {selected.audit_sensitivity||'-'}</div>
                  <div><b>Procurement:</b> {selected.procurement_rigor||'-'}</div>
                  <div><b>Mission Tags:</b> {(selected.mission_tags||[]).join(', ')||'-'}</div>
                </div>
                <div className="mt-4 flex gap-2">
                  <button className="btn-cyber" onClick={async()=>{
                    const res = await fetch(`${API_BASE}/funding/toggle`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: selected.id, enabled: !(selected.enabled!==false)})});
                    if (res.ok){ setSelected(null); const d = await fetch(`${API_BASE}/funding/rank${onlyEnabled ? '?enabled=true':''}`).then(r=>r.json()); setItems(d.programs||[]); }
                  }}>{selected.enabled!==false ? 'Disable' : 'Enable'}</button>
                  <a className="btn-cyber" style={{background:'#6b7280'}} target="_blank" href={selected.source_url||'#'} rel="noreferrer">Source</a>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>


        <div className="mt-6 text-xs text-[color:var(--muted-400)]">
          <p>Notes:</p>
          <ul className="list-disc ml-5">
            <li>Usefulness and cost-to-service scores can be adjusted via the sliders above.</li>
            <li>REAP is listed but disabled by default. You can show disabled programs via the toggle above.</li>
            <li>Use the Save toggles action to persist current enabled flags to disk.</li>
          </ul>
        </div>
      </div>
      </AppShell>
    </>
  );
}

