import React from 'react';
import { AlertTriangle, CheckCircle, XCircle, Cpu, MemoryStick, Activity, Zap } from 'lucide-react';
import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts';

const SEVERITY_COLORS = {
  0: '#22c55e', // INFO  - green
  1: '#f59e0b', // WARN  - amber
  2: '#ef4444', // ERROR - red
  3: '#7c3aed', // CRITICAL - purple
};

function RiskGauge({ score }) {
  const pct = Math.round((score ?? 0) * 100);
  const color = pct < 30 ? '#22c55e' : pct < 60 ? '#f59e0b' : pct < 80 ? '#ef4444' : '#7c3aed';
  const data = [{ value: pct, fill: color }, { value: 100 - pct, fill: '#1e293b' }];

  return (
    <div className="relative w-16 h-16">
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          innerRadius="60%"
          outerRadius="100%"
          data={data}
          startAngle={90}
          endAngle={-270}
        >
          <RadialBar dataKey="value" cornerRadius={4} background={{ fill: '#1e293b' }} />
        </RadialBarChart>
      </ResponsiveContainer>
      <span
        className="absolute inset-0 flex items-center justify-center text-xs font-bold"
        style={{ color }}
      >
        {pct}%
      </span>
    </div>
  );
}

function MetricBar({ value, max, color, label }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="mb-1">
      <div className="flex justify-between text-xs text-slate-400 mb-0.5">
        <span>{label}</span>
        <span>{value?.toFixed(1)}</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function NodeCard({ name, health, riskScore, onHeal }) {
  const isAlive = health?.is_alive ?? false;
  const cpu = health?.cpu_percent ?? 0;
  const mem = health?.memory_percent ?? 0;
  const latency = health?.mean_latency_ms ?? 0;
  const severity = health?.severity ?? 0;
  const borderColor = isAlive ? SEVERITY_COLORS[severity] : '#475569';
  const shortName = name.replace(/^\//, '').replace(/_/g, ' ');

  return (
    <div
      className="rounded-xl p-4 bg-slate-800 transition-all duration-300 hover:scale-[1.01]"
      style={{ border: `1px solid ${borderColor}30`, boxShadow: `0 0 12px ${borderColor}18` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          {isAlive
            ? <CheckCircle size={14} color="#22c55e" />
            : <XCircle size={14} color="#ef4444" />}
          <span className="text-sm font-semibold text-slate-100 truncate capitalize">{shortName}</span>
        </div>
        <RiskGauge score={riskScore} />
      </div>

      {/* Metrics */}
      <div className="space-y-1 mb-3">
        <MetricBar value={cpu} max={100} color="#3b82f6" label="CPU %" />
        <MetricBar value={mem} max={100} color="#8b5cf6" label="MEM %" />
        <MetricBar value={latency} max={500} color="#06b6d4" label="Latency ms" />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between">
        <span
          className="text-xs px-2 py-0.5 rounded-full font-medium"
          style={{ backgroundColor: `${SEVERITY_COLORS[severity]}20`, color: SEVERITY_COLORS[severity] }}
        >
          {['Healthy', 'Warning', 'Error', 'Critical'][severity]}
        </span>
        {severity >= 1 && (
          <button
            onClick={() => onHeal(name)}
            className="text-xs px-2 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors flex items-center gap-1"
          >
            <Zap size={10} /> Heal
          </button>
        )}
      </div>
    </div>
  );
}
