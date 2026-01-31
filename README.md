# iQuHack 2026 - Quantum Entanglement Distillation Game

A competitive quantum networking game where players build subgraphs by claiming edges through entanglement distillation.

---

## 🚀 Quick Start

### 1. Install Dependencies (in a virtual environment)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Playing!

Open and run `demo.ipynb` in Jupyter (or VS Code):

```bash
jupyter notebook demo.ipynb
```

The notebook contains:
- ✅ Complete tutorial (5-minute quick start)
- ✅ All SDK features with examples
- ✅ Visualization demos
- ✅ Full game workflow with strategies
- ✅ Advanced tips and tricks

---

## 📖 Game Overview

### Objective

Build a quantum network subgraph to maximize your score by claiming edges through entanglement distillation.

### How It Works

1. **Register** with a unique player ID and location
2. **Select** a starting node from candidates
3. **Design** quantum distillation circuits
4. **Claim edges** by beating fidelity thresholds
5. **Earn points** by collecting nodes with utility qubits
6. **Manage** your limited bell pair budget
7. **Compete** on the leaderboard!

### Key Mechanics

- 🌐 **Graph**: Quantum network with nodes (utility qubits) and edges (entanglement links)
- ⚡ **Distillation**: Submit circuits (circuit_a, circuit_b) to purify noisy Bell pairs
- 🎯 **Thresholds**: Achieve fidelity ≥ base_threshold to claim an edge
- 💰 **Budget**: Limited bell pairs to use for distillation attempts
- 🏆 **Scoring**: Sum of utility qubits from owned nodes (vertices with top claim strength)

---

## 📦 Repository Structure

```
iQuHack2026-private/
├── demo.ipynb             # 👈 START HERE - Complete interactive tutorial
├── client.py              # GameClient class (API wrapper)
├── visualization.py       # GraphTool class (focused rendering)
├── entanglement_utils.py  # Quantum utilities for circuit testing
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

---

## 🎮 SDK Features

### GameClient - Easy API Access

```python
from client import GameClient

client = GameClient()

# One-line registration
result = client.register("player_id", "Name", location="remote")

# Auto-select best starting node
client.auto_select_best_starting_node(candidates, prefer="utility")

# Get claimable edges automatically
claimable = client.get_claimable_edges()

# Quick access to player state
budget = client.get_budget()
score = client.get_score()
owned_nodes = client.get_owned_nodes()

# Pretty-printed status
client.print_status()
```

### GraphTool - Focused Visualization

```python
from visualization import GraphTool

viz = GraphTool(client.get_cached_graph())
owned = set(client.get_owned_nodes())

# Shows only nodes within 2 hops of your owned nodes
viz.render_focused(owned, radius=2)

# Text summary (focused view)
viz.print_summary(owned, focused=True)
```

**Why focused?** Large graphs become cluttered. Focused view shows only relevant nodes (~50-95% reduction), making it much easier to see claimable edges!

---

## 🔑 API Endpoints

Base URL: `https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app`

### Public (No Auth)
- `GET /v1/health` - Server health check
- `GET /v1/graph` - Get graph structure
- `GET /v1/leaderboard` - Get player rankings
- `GET /v1/visualization` - D3.js visualization

### Protected (Bearer Token Required)
- `POST /v1/register` - Register player (returns api_token)
- `POST /v1/select_starting_node` - Choose starting node
- `POST /v1/claim_edge` - Submit distillation circuits
- `GET /v1/status/{player_id}` - Get player status
- `POST /v1/restart` - Reset progress

**Authentication**: Bearer token issued at registration, automatically stored by SDK.

---

## 🎯 Game Strategy Tips

### 1. Starting Node Selection
- **High utility** → More immediate points
- **High bonus** → More budget for future claims
- **Balanced** → Best of both worlds

### 2. Edge Claiming Strategy
- **Greedy**: Target easiest edges first (low difficulty)
- **High-value**: Target edges leading to high-utility nodes
- **Budget-conscious**: Use minimal bell pairs per attempt

### 3. Circuit Design
- Start simple (H-CNOT gates)
- Study `entanglement_utils.py` for testing locally
- Research protocols: BBPSSW, DEJMPS, etc.
- More bell pairs → better fidelity, but costs more budget

### 4. Competition
- Monitor leaderboard frequently
- Vertices have capacity limits - only top N players by claim strength earn rewards
- Budget management is key - don't waste attempts!

---

## 🔬 Advanced: Entanglement Utilities

Test your distillation circuits locally before submitting:

```python
import entanglement_utils as eq
import numpy as np

# Generate noisy Bell state
bell_coeffs = np.array([0.9, 0.05, 0.05, 0.0])
fidelity = eq.calculate_phi_plus_fidelity(bell_coeffs)

# Test with multiple pairs
initial_pairs = eq.generate_initial_noisy_pairs(nbells=2, bell_coeffs=bell_coeffs)

# Design circuits, simulate, measure fidelity
# ... then submit to server when ready!
```

---

## 📚 Documentation

### In This Repository
- **demo.ipynb** - Interactive tutorial with all features
- **client.py** - Full docstrings on all methods
- **visualization.py** - Full docstrings on all methods
- **entanglement_utils.py** - Quantum utility functions

### Python Help
```python
help(client.get_claimable_edges)
help(viz.render_focused)
```

---

## 🏆 Winning the Game

1. **Design efficient distillation circuits**
   - Beat fidelity thresholds with minimal bell pairs
   - Research quantum error correction and purification

2. **Optimize your path**
   - Plan which edges to claim for maximum utility gain
   - Consider graph topology and vertex capacity constraints

3. **Manage your budget**
   - Don't waste bell pairs on low-probability claims
   - Balance risk vs. reward

4. **Stay competitive**
   - Monitor leaderboard
   - Adapt strategy based on other players

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Invalid token" errors
Re-register or reconnect with saved token:
```python
client = GameClient(api_token="your-saved-token")
```

### Visualization not showing
Make sure matplotlib is installed:
```bash
pip install matplotlib
```

### Game questions
See `demo.ipynb` for comprehensive examples!

---

## 🤝 Support

- **Issues**: Open an issue in this repository
- **Questions**: Check `demo.ipynb` first - it covers most scenarios
- **Bugs**: Report with reproduction steps

---

## 📄 License

This is challenge code for iQuHack 2026. Use for educational and competition purposes.

---

## 🎉 Good Luck!

Remember: The best quantum engineer wins! Design efficient circuits, manage your budget wisely, and claim strategically.

**May the highest fidelity win!** 🚀⚛️

---

**Next Step**: Open `demo.ipynb` and start playing! 🎮
