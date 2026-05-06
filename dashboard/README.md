# ROS2 Guardian Dashboard

Real-time monitoring UI for the Guardian platform. Built with React 18 + Recharts + Tailwind CSS.

## Quick Start

```bash
cd dashboard
npm install
npm start          # dev server → http://localhost:3000
npm run build      # production build → build/
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REACT_APP_WS_URL` | `ws://localhost:8765` | Guardian bridge WebSocket URL |

## Architecture

```
App.jsx
├── hooks/useGuardianWS.js    ← WebSocket connection + message routing
├── components/NodeCard.jsx   ← Per-node health + risk gauge + heal button
├── components/AlertFeed.jsx  ← Scrollable alert list with severity badges
└── components/HealingLog.jsx ← Audit trail of healing actions
```

## Data Sources

All data comes from the `dashboard_bridge_node` WebSocket server which relays:

| ROS2 Topic | Dashboard Effect |
|---|---|
| `/guardian/metrics` | Updates node cards |
| `/guardian/alerts` | Appends to alert feed |
| `/guardian/risk_scores` | Updates risk gauges |
| `/guardian/healing_events` | Appends to healing log |

## Commands

The dashboard can send commands back to Guardian:

```json
{ "cmd": "trigger_healing", "node_name": "/some_node" }
```
