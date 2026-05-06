import { useState, useEffect, useRef, useCallback } from 'react';

const WS_RECONNECT_DELAY_MS = 3000;
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8765';

export function useGuardianWS() {
  const [connected, setConnected] = useState(false);
  const [nodeHealth, setNodeHealth] = useState({});   // { nodeName: NodeHealth }
  const [alerts, setAlerts] = useState([]);            // SystemAlert[]
  const [riskScores, setRiskScores] = useState({});    // { nodeName: float }
  const [healingEvents, setHealingEvents] = useState([]); // HealingAction[]
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        dispatch(msg);
      } catch { /* ignore malformed */ }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, WS_RECONNECT_DELAY_MS);
    };

    ws.onerror = () => ws.close();
  }, []);

  const dispatch = (msg) => {
    switch (msg.topic) {
      case '/guardian/metrics':
        setNodeHealth(prev => ({ ...prev, [msg.data.node_name]: msg.data }));
        break;
      case '/guardian/alerts':
        setAlerts(prev => [msg.data, ...prev].slice(0, 100));
        break;
      case '/guardian/risk_scores':
        setRiskScores(prev => ({ ...prev, [msg.data.node_name]: msg.data.anomaly_score }));
        break;
      case '/guardian/healing_events':
        setHealingEvents(prev => [msg.data, ...prev].slice(0, 50));
        break;
      default:
        break;
    }
  };

  const triggerHealing = useCallback((nodeName) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ cmd: 'trigger_healing', node_name: nodeName }));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected, nodeHealth, alerts, riskScores, healingEvents, triggerHealing };
}
