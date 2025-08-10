"use client";

import dynamic from 'next/dynamic';
const Line = dynamic(() => import('react-chartjs-2').then(m => m.Line), { ssr: false });
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

export default function MiniSparkline({ series = [], height = 36, color = 'hsl(var(--primary))' }) {
  const data = {
    labels: series.map((_, i) => i + 1),
    datasets: [
      {
        data: series,
        borderColor: color,
        backgroundColor: 'rgba(0,0,0,0)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.35,
        fill: false,
      },
    ],
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { display: false },
      y: { display: false },
    },
    elements: { point: { radius: 0 } },
  };
  return (
    <div style={{ height }}>
      <Line data={data} options={options} />
    </div>
  );
}

