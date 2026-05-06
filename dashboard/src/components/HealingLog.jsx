import React from 'react';
import { CheckCircle2, XCircle, Clock, Wrench } from 'lucide-react';

export default function HealingLog({ events }) {
  if (!events.length) {
    return (
      <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-sm">
        <Wrench size={22} className="mb-2 opacity-40" />
        No healing actions yet
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
      {events.map((ev, i) => {
        const success = ev.status === 'success' || ev.status === 0;
        return (
          <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/60">
            {success
              ? <CheckCircle2 size={14} className="text-green-400 mt-0.5 shrink-0" />
              : <XCircle size={14} className="text-red-400 mt-0.5 shrink-0" />}
            <div className="flex-1 min-w-0">
              <div className="flex justify-between">
                <span className="text-xs font-semibold text-slate-200 capitalize">
                  {(ev.action_name || 'unknown').replace(/_/g, ' ')}
                </span>
                <span className="text-xs text-slate-500 flex items-center gap-1">
                  <Clock size={10} />
                  {ev.duration_ms != null ? `${ev.duration_ms}ms` : '—'}
                </span>
              </div>
              <p className="text-xs text-slate-400 truncate mt-0.5">{ev.message}</p>
              {ev.attempt != null && (
                <span className="text-xs text-slate-600">attempt #{ev.attempt}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
