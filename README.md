## iQuHack 2026 — Entanglement Distillation Game (IonQ)

This repo contains our **hackathon toolkit** for the entanglement distillation game: a small Python SDK, a notebook “control center”, and visual + automated strategy helpers.

### Our approach (what we built + why)

- **Circuit toolkit first**: we implemented and iterated on practical OpenQASM 3 / Qiskit distillation circuits (with explicit `flag_bit` post-selection) so we can reliably convert “\(N\) noisy pairs” → “one higher-fidelity pair” per edge type. See `distillation_circuits.py`.
- **Optimize what the game rewards**: edge contribution is driven by **fidelity × success_probability**, so we target circuits/parameters that maximize \(F \cdot p\) while still meeting the edge’s threshold.
- **Expansion strategy**: we prioritize reachable, high-value nodes (utility qubits + bonus bell pairs) while managing difficulty and failure streaks. See `greedy_auto_viz.py`.
- **Fast feedback loops**: we built UIs to reduce iteration time—manual interactive play, and an auto-running greedy visualizer. See `interactive_viz.py` and `web_client.html`.

### Quickstart

#### Python (recommended: notebook workflow)

```bash
cd 2026-IonQ
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Open `demo.ipynb` (VS Code or Jupyter). It includes **session save/load** via `session.json` so you don’t have to re-register every time.

#### Browser UI (quick manual play + map)

```bash
cd 2026-IonQ
python proxy_server.py
```

Then open `http://localhost:5173/web_client.html`.

### Repository structure (what to open first)

```text
2026-IonQ/
  demo.ipynb              Notebook “home base” (register, play, iterate)
  client.py               GameClient (API wrapper + claim helpers)
  distillation_circuits.py Circuit library / templates
  visualization.py        Static graph utilities (NetworkX/Matplotlib)
  interactive_viz.py       Interactive map + “click edge, attack” workflow
  greedy_auto_viz.py       Auto-running greedy strategy visualizer
  web_client.html          Standalone web client UI (uses a proxy for CORS)
  proxy_server.py          Local static server + CORS proxy for web_client.html
  api/proxy.js             Vercel-style serverless proxy (optional deployment)
  game_handbook.md         Game rules (LOCC constraints, scoring, etc.)
  requirements.txt         Python dependencies
  session.json             Local saved token/session (do not commit)
```

### Configuration notes

- **Upstream server**: defaults to `https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app`.
  - Python: pass `base_url=...` to `GameClient(...)`.
  - Web/proxy: set `UPSTREAM_BASE_URL` if you need to point elsewhere.
