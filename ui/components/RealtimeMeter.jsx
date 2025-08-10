"use client";

import { useEffect, useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { useMetricsStore } from "@/lib/metricsStore";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip);

// Simple 24-hour activity meter. Pass a "source" async function that returns the latest value.
export default function RealtimeMeter({ title = 'Throughput', unit = '', signal, apiBase = (typeof window!=='undefined' && process.env.NEXT_PUBLIC_API_BASE) ? process.env.NEXT_PUBLIC_API_BASE : 'http://localhost:8000/api', intervalMs = 5000, color = '#22d3ee' }) {
  const [series, setSeries] = useState(() => Array.from({ length: 24 }, () => 0));
  const push = useMetricsStore?.getState?.().pushMeterSample;

  useEffect(() => {
    let timer;
    const tick = async () => {
      try {
        let val;
        if (signal) {
          const res = await fetch(`${apiBase}/metrics/realtime?signal=${encodeURIComponent(signal)}`);
          if (res.ok) {
            const j = await res.json();
            val = j.value;
          }
        }
        if (val === undefined) val = Math.floor(Math.random() * 100);
        setSeries((s) => [...s.slice(1), val]);
        try { if (push && signal) push(signal, val, 24); } catch {}
      } catch {}
      timer = setTimeout(tick, intervalMs);
    };
    tick();
    return () => clearTimeout(timer);
  }, [signal, apiBase, intervalMs]);

  const data = useMemo(() => ({
    labels: series.map((_, i) => i),
    datasets: [{ data: series, borderColor: color, backgroundColor: 'rgba(0,0,0,0)', pointRadius: 0, tension: 0.35, fill: false }],
  }), [series, color]);

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: { x: { display: false }, y: { display: false } },
    elements: { point: { radius: 0 } },
  }), []);

  const latest = series[series.length - 1] ?? 0;

  return (
    <div className="rounded-lg border p-5">
      <div className="flex items-baseline justify-between">
        <div className="text-sm text-muted-foreground">{title}</div>
        <div className="text-xs text-muted-foreground">Last 24h</div>
      </div>
      <div className="text-2xl font-semibold">{latest.toLocaleString()} {unit}</div>
      <div className="h-16 mt-2">
        <Line data={data} options={options} />
      </div>
    </div>
  );
}

