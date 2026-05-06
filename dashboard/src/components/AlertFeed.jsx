import React from 'react';
import { AlertTriangle, Info, XOctagon, Skull } from 'lucide-react';

const ICONS = [Info, AlertTriangle, XOctagon, Skull];
const COLORS = ['#22c55e', '#f59e0b', '#ef4444', '#7c3aed'];
const LABELS = ['INFO', 'WARN', 'ERROR', 'CRIT'];

function timeAgo(tsMs) {
  const diff = (Date.now() - tsMs) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return `${Math.round(diff / 3600)}h ago`;
}

export default function AlertFeed({ alerts }) {
  if (!alerts.length) {
    return (
      <div className="flex flex-col items-center justify-center h-40 text-slate-500 text-sm">
        <Info size={24} className="mb-2 opacity-40" />
        No alerts — system healthy
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-80 overflow-y-auto pr-1 scrollbar-thin">
      {alerts.map((alert, i) => {
        const sev = alert.severity ?? 0;
        const Icon = ICONS[sev];
        const color = COLORS[sev];
        const ts = alert.header?.stamp
          ? alert.header.stamp.sec * 1000
          : Date.now() - i * 5000;

        return (
          <div
            key={i}
            className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/60"
            style={{ borderLeft: `3px solid ${color}` }}
          >
            <Icon size={14} color={color} className="mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex justify-between items-center mb-0.5">
                <span
                  className="text-xs font-bold uppercase tracking-wide"
                  style={{ color }}
                >
                  {LABELS[sev]}
                </span>
                <span className="text-xs text-slate-500">{timeAgo(ts)}</span>
              </div>
              <p className="text-xs text-slate-300 truncate">
                <span className="text-slate-400 font-medium">{alert.node_name}</span>
                {' — '}
                {alert.message}
              </p>
              {alert.anomaly_score != null && (
                <div className="mt-1 flex items-center gap-1">
                  <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${alert.anomaly_score * 100}%`, backgroundColor: color }}
                    />
                  </div>
                  <span className="text-xs text-slate-500">
                    {(alert.anomaly_score * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
