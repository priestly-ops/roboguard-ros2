import React, { useState } from 'react';
import { Shield, Wifi, WifiOff, RefreshCw, Activity } from 'lucide-react';
import { useGuardianWS } from './hooks/useGuardianWS';
import NodeCard from './components/NodeCard';
import AlertFeed from './components/AlertFeed';
import HealingLog from './components/HealingLog';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts';

const CHART_COLORS = [
  '#3b82f6', '#8b5cf6', '#06b6d4', '#22c55e',
  '#f59e0b', '#ef4444', '#ec4899', '#14b8a6'
];

function Section({ title, children, className = '' }) {
  return (
    <div className={`rounded-xl bg-slate-900 border border-slate-700/50 p-4 ${className}`}>
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
        {title}
      </h2>
      {children}
    </div>
  );
}

function StatusBadge({ connected }) {
  return (
    <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
      connected ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
    }`}>
      {connected ? <Wifi size={12} /> : <WifiOff size={12} />}
      {connected ? 'Live' : 'Disconnected'}
    </div>
  );
}

// Build rolling CPU chart data from nodeHealth
function useCpuHistory(nodeHealth) {
  const [history, setHistory] = React.useState([]);

  React.useEffect(() => {
    if (!Object.keys(nodeHealth).length) return;
    setHistory(prev => {
      const entry = { t: new Date().toLocaleTimeString() };
      for (const [name, h] of Object.entries(nodeHealth)) {
        const key = name.replace(/^\//, '').slice(0, 12);
        entry[key] = h.cpu_percent?.toFixed(1) ?? 0;
      }
      return [...prev.slice(-29), entry];
    });
  }, [nodeHealth]);

  return history;
}

export default function App() {
  const { connected, nodeHealth, alerts, riskScores, healingEvents, triggerHealing } = useGuardianWS();
  const cpuHistory = useCpuHistory(nodeHealth);
  const nodeNames = Object.keys(nodeHealth);
  const criticalCount = alerts.filter(a => a.severity >= 2).length;
  const healedCount = healingEvents.filter(e => e.status === 'success').length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 font-sans">
      {/* ── Header ── */}
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-blue-600/20 border border-blue-500/30">
            <Shield size={22} className="text-blue-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">ROS2 Guardian</h1>
            <p className="text-xs text-slate-500">AI-powered robot health monitor</p>
          </div>
        </div>
        <StatusBadge connected={connected} />
      </header>

      {/* ── KPI strip ── */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {[
          { label: 'Nodes', value: nodeNames.length, icon: <Activity size={14} />, color: 'blue' },
          { label: 'Critical', value: criticalCount, icon: <Activity size={14} />, color: criticalCount ? 'red' : 'green' },
          { label: 'Healed', value: healedCount, icon: <RefreshCw size={14} />, color: 'purple' },
        ].map(({ label, value, icon, color }) => (
          <div key={label} className={`rounded-xl bg-${color}-500/10 border border-${color}-500/20 p-3 text-center`}>
            <div className={`text-2xl font-bold text-${color}-400`}>{value}</div>
            <div className="text-xs text-slate-500 mt-0.5 uppercase tracking-wide">{label}</div>
          </div>
        ))}
      </div>

      {/* ── Main grid ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Node cards */}
        <Section title={`Nodes (${nodeNames.length})`} className="lg:col-span-2">
          {nodeNames.length === 0 ? (
            <div className="text-center py-10 text-slate-500 text-sm">
              Waiting for Guardian metrics…
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {nodeNames.map((name, i) => (
                <NodeCard
                  key={name}
                  name={name}
                  health={nodeHealth[name]}
                  riskScore={riskScores[name]}
                  onHeal={triggerHealing}
                />
              ))}
            </div>
          )}
        </Section>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          <Section title="Recent Alerts">
            <AlertFeed alerts={alerts} />
          </Section>
          <Section title="Healing Log">
            <HealingLog events={healingEvents} />
          </Section>
        </div>

        {/* CPU trend chart */}
        <Section title="CPU Trend (30s)" className="lg:col-span-3">
          {cpuHistory.length < 2 ? (
            <div className="h-32 flex items-center justify-center text-slate-500 text-sm">
              Collecting data…
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={cpuHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#64748b' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} unit="%" />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {nodeNames.map((name, i) => {
                  const key = name.replace(/^\//, '').slice(0, 12);
                  return (
                    <Line
                      key={name}
                      type="monotone"
                      dataKey={key}
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      strokeWidth={1.5}
                      dot={false}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          )}
        </Section>
      </div>

      <footer className="mt-6 text-center text-xs text-slate-600">
        ROS2 Guardian · Apache 2.0 · Bridge ws://localhost:8765
      </footer>
    </div>
  );
}
