"use client";

import { useMetricsStore } from "@/lib/metricsStore";

export default function Rightbar(){
  const kpis = useMetricsStore((s)=>s.kpis);
  return (
    <aside className="w-full xl:w-72 cyber-panel rounded-xl h-[calc(100vh-6rem)] p-4 sticky top-24">
      <div className="text-[color:var(--text-100)] font-semibold mb-3">KPIs</div>
      <div className="space-y-3 text-sm">
        <div className="rounded-lg border p-3">
          <div className="text-xs text-muted-foreground">Food Bank</div>
          <div className="text-xl font-semibold">{(kpis.food_bank||0).toLocaleString()}</div>
        </div>
        <div className="rounded-lg border p-3">
          <div className="text-xs text-muted-foreground">Mill</div>
          <div className="text-xl font-semibold">{(kpis.mill||0).toLocaleString()}</div>
        </div>
        <div className="rounded-lg border p-3">
          <div className="text-xs text-muted-foreground">Op Cost (mo)</div>
          <div className="text-xl font-semibold">${(kpis.op_cost_monthly||0).toLocaleString()}</div>
        </div>
        <div className="rounded-lg border p-3">
          <div className="text-xs text-muted-foreground">Op Revenue (mo)</div>
          <div className="text-xl font-semibold">${(kpis.op_revenue_monthly||0).toLocaleString()}</div>
        </div>
      </div>
    </aside>
  );
}

