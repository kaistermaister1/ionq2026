# =========================
# Run D1 (bit-flip) optimal N=2 circuit on a specific SOURCE <-> TARGET edge and print stats
# Copy/paste this whole cell
# =========================

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit import qasm3

def get_d1_opt_circuit_N2():
    """
    D1: rho = F|Phi+><Phi+| + (1-F)|Psi+><Psi+|  (bit-flip only).
    Optimal N=2 recurrence step:
      - output pair is (q1,q2)
      - sacrificial pair is (q0,q3)
      - bilateral parity check: CX(q1->q0) on Alice, CX(q2->q3) on Bob
      - measure sacrificial qubits
      - flag = c0 XOR c1 (keep iff flag=0)
    """
    qr = QuantumRegister(4, "q")          # 2N = 4 qubits
    cr = ClassicalRegister(3, "c")        # c[0], c[1] syndromes; c[2] flag
    qc = QuantumCircuit(qr, cr)

    # Bilateral parity check (within each side only)
    qc.cx(qr[1], qr[0])   # Alice: q1 -> q0
    qc.cx(qr[2], qr[3])   # Bob:   q2 -> q3

    # Measure sacrificial qubits
    qc.measure(qr[0], cr[0])
    qc.measure(qr[3], cr[1])

    # Flag = mismatch = c0 XOR c1  (postselect flag_bit==0)
    qc.store(cr[2], expr.bit_xor(cr[0], cr[1]))

    return qc


def run_d1_n2_on_edge(client, source_city: str, target_city: str):
    """
    Runs the D1 N=2 optimal circuit on the specific claimable edge SOURCE <-> TARGET
    and prints the same stats format you requested.
    """
    claimable = client.get_claimable_edges()
    want = {source_city, target_city}
    target_edge = next((e for e in claimable if set(e["edge_id"]) == want), None)

    if target_edge is None:
        print(f"Edge {source_city} <-> {target_city} is NOT currently claimable.")
        print("Some claimable edges (first 12):")
        for e in claimable[:12]:
            print(f"  {tuple(e['edge_id'])} | diff={e['difficulty_rating']} | thr={e['base_threshold']}")
        return None

    edge_id = tuple(target_edge["edge_id"])
    diff = int(target_edge.get("difficulty_rating", -1))
    thr  = float(target_edge.get("base_threshold", 0.0))

    print(f"Attacking edge: {edge_id} | diff={diff} | thr={thr:.2f}")
    if diff != 1:
        print("WARNING: This is not a D1 edge. This circuit is tuned for D1 (Phi+/Psi+ mixture).")

    result = client.claim_edge(
        edge=edge_id,
        circuit=get_d1_opt_circuit_N2(),
        flag_bit=2,        # <-- c[2]
        num_bell_pairs=2   # <-- N=2
    )

    if not result.get("ok"):
        print("FAILED:", result.get("error", {}).get("code"), "-", result.get("error", {}).get("message"))
        return result

    data = result["data"]
    F = float(data.get("fidelity", 0.0))
    p = float(data.get("success_probability", 0.0))

    print(f"Success: {data.get('success')}")
    print(f"Fidelity: {F:.4f} (threshold: {float(data.get('threshold', 0.0)):.4f})")
    print(f"Success probability: {p:.4f}")
    print(f"Claim strength added (f*p): {F*p:.4f}")

    return result

# =========================
# Run D2 circuit on a specific SOURCE <-> TARGET edge and print stats
# Copy/paste this whole cell
# =========================

def get_d2_circuit():
    # N=2 -> 4 qubits total, 3 classical bits (c2 is flag)
    qr = QuantumRegister(4, "q")
    cr = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qr, cr)

    # Convert phase flips -> bit flips
    qc.h(qr)

    # Bilateral parity check (output pair q1,q2 controls sacrificial q0,q3)
    qc.cx(qr[1], qr[0])   # Alice side
    qc.cx(qr[2], qr[3])   # Bob side

    # Measure sacrificial qubits
    qc.measure(qr[0], cr[0])
    qc.measure(qr[3], cr[1])

    # Restore basis on output pair
    qc.h(qr[1])
    qc.h(qr[2])

    # Flag = c0 XOR c1 (keep if flag_bit == 0)
    qc.store(cr[2], expr.bit_xor(cr[0], cr[1]))

    return qc


def run_d2_on_edge(client, source_city: str, target_city: str):
    # Find that exact edge among claimable edges
    claimable = client.get_claimable_edges()
    want = {source_city, target_city}
    target_edge = next((e for e in claimable if set(e["edge_id"]) == want), None)

    if target_edge is None:
        print(f"Edge {source_city} <-> {target_city} is NOT currently claimable.")
        print("Some claimable edges (first 12):")
        for e in claimable[:12]:
            print(f"  {tuple(e['edge_id'])} | diff={e['difficulty_rating']} | thr={e['base_threshold']}")
        return None

    edge_id = tuple(target_edge["edge_id"])

    result = client.claim_edge(
        edge=edge_id,
        circuit=get_d2_circuit(),
        flag_bit=2,        # <-- c[2]
        num_bell_pairs=2   # <-- N=2
    )

    if not result.get("ok"):
        print("FAILED:", result.get("error", {}).get("code"), "-", result.get("error", {}).get("message"))
        return result

    data = result["data"]
    F = float(data.get("fidelity", 0.0))
    p = float(data.get("success_probability", 0.0))

    print(f"Success: {data.get('success')}")
    print(f"Fidelity: {F:.4f} (threshold: {float(data.get('threshold', 0.0)):.4f})")
    print(f"Success probability: {p:.4f}")
    print(f"Claim strength added (f*p): {F*p:.4f}")

    return result


# =========================
# D3 optimal N=3 (Phi+/Psi+) circuit + runner (QASM3, same style as your working D2/D4)
# Copy/paste this whole cell
# =========================

def get_d3_opt_circuit_N3():
    """
    D3 noise (your measured model): rho = 0.75|Phi+><Phi+| + 0.25|Psi+><Psi+|
    This is "bit-flip-type" (Psi+ corresponds to an X error relative to Phi+).

    N=3 protocol (recurrence):
      - 2 sacrificial pairs + 1 output pair
      - Do bilateral parity checks from the output pair onto BOTH sacrificial pairs
      - Measure sacrificial qubits and postselect that BOTH parity syndromes are 0
      - Keep output pair (q2,q3)

    Pairing for N=3 (2N=6, outside-in):
      Pair0: (q0, q5)  sacrificial
      Pair1: (q1, q4)  sacrificial
      Pair2: (q2, q3)  output  <-- keep

    Expected (theory) for F=0.75:
      p_succ = F^3 + (1-F)^3 = 0.4375
      F_out  = F^3 / (F^3 + (1-F)^3) ≈ 0.9642857
    """

    qr = QuantumRegister(6, "q")          # 2N = 6 qubits
    cr = ClassicalRegister(7, "c")        # c0..c3 meas, c4 s0, c5 s1, c6 flag
    qc = QuantumCircuit(qr, cr)

    # --- Bilateral parity checks from output pair (q2,q3) -> sacrificial pairs ---
    # check output vs Pair0 (q0,q5)
    qc.cx(qr[2], qr[0])   # Alice side
    qc.cx(qr[3], qr[5])   # Bob side

    # check output vs Pair1 (q1,q4)
    qc.cx(qr[2], qr[1])   # Alice side
    qc.cx(qr[3], qr[4])   # Bob side

    # --- Measure sacrificial qubits ---
    qc.measure(qr[0], cr[0])  # Alice (pair0)
    qc.measure(qr[1], cr[1])  # Alice (pair1)
    qc.measure(qr[5], cr[2])  # Bob   (pair0)
    qc.measure(qr[4], cr[3])  # Bob   (pair1)

    # --- Build syndromes then flag with parser-friendly simple assignments ---
    qc.store(cr[4], expr.bit_xor(cr[0], cr[2]))  # s0 = c0 ^ c2
    qc.store(cr[5], expr.bit_xor(cr[1], cr[3]))  # s1 = c1 ^ c3
    qc.store(cr[6], expr.bit_or(cr[4], cr[5]))   # flag = s0 | s1  (keep iff flag == 0)

    return qc


def run_d3_n3_on_edge(client, source_city: str, target_city: str, verbose_qasm_tail: int = 0):
    """
    Finds the specific claimable edge SOURCE <-> TARGET, submits D3 N=3 circuit,
    and prints stats in your preferred format.
    """
    claimable = client.get_claimable_edges()
    want = {source_city, target_city}
    target_edge = next((e for e in claimable if set(e["edge_id"]) == want), None)

    if target_edge is None:
        print(f"Edge {source_city} <-> {target_city} is NOT currently claimable.")
        print("Some claimable edges (first 12):")
        for e in sorted(claimable, key=lambda x: (x["difficulty_rating"], x["base_threshold"]))[:12]:
            print(f"  {tuple(e['edge_id'])} | diff={e['difficulty_rating']} | thr={e['base_threshold']}")
        return None

    edge_id = tuple(target_edge["edge_id"])
    diff = int(target_edge.get("difficulty_rating", -1))
    thr  = float(target_edge.get("base_threshold", 0.0))

    print(f"Attacking edge: {edge_id} | diff={diff} | thr={thr:.2f}")
    if diff != 3:
        print("WARNING: This circuit is tuned for D3 (Phi+/Psi+ mixture).")

    circuit = get_d3_opt_circuit_N3()

    if verbose_qasm_tail > 0:
        q = qasm3.dumps(circuit).splitlines()
        print("\n=== QASM3 tail preview ===")
        print("\n".join(q[-verbose_qasm_tail:]))
        print("=========================\n")

    # IMPORTANT:
    # - N=3
    # - flag is c[6]
    result = client.claim_edge(
        edge=edge_id,
        circuit=circuit,
        flag_bit=6,
        num_bell_pairs=3
    )

    if not result.get("ok"):
        print("FAILED:", result.get("error", {}).get("code"), "-", result.get("error", {}).get("message"))
        return result

    data = result["data"]
    F = float(data.get("fidelity", 0.0))
    p = float(data.get("success_probability", 0.0))

    print(f"Success: {data.get('success')}")
    print(f"Fidelity: {F:.4f} (threshold: {float(data.get('threshold', 0.0)):.4f})")
    print(f"Success probability: {p:.4f}")
    print(f"Claim strength added (f*p): {F*p:.4f}")

    return result


# =========================
# D4 optimal N=3 submit helper (QASM3, no fallback)
# Paste this whole cell, then set SOURCE/TARGET at bottom and run.
# =========================

def get_d4_opt_circuit_N3():
    """
    D4 (phase-flip-only) N=3 protocol with QASM3-safe classical logic.
    Uses 2 sacrificial pairs + 1 output pair.
    Postselect on flag_bit = c[6] == 0.
    """
    qr = QuantumRegister(6, "q")          # 2N = 6 qubits
    cr = ClassicalRegister(7, "c")        # need temps: c[4], c[5], flag c[6]
    qc = QuantumCircuit(qr, cr)

    # 1) Convert phase flips -> bit flips
    for i in range(6):
        qc.h(qr[i])

    # Pairing for N=3:
    # sacrificial: (q0,q5) and (q1,q4)
    # output:      (q2,q3)  <-- keep

    # 2) Two bilateral parity checks from output -> sacrificial
    qc.cx(qr[2], qr[0])  # Alice
    qc.cx(qr[3], qr[5])  # Bob
    qc.cx(qr[2], qr[1])  # Alice
    qc.cx(qr[3], qr[4])  # Bob

    # 3) Measure sacrificial qubits
    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])
    qc.measure(qr[5], cr[2])
    qc.measure(qr[4], cr[3])

    # 4) Restore basis on output pair (so fidelity is measured vs Phi+)
    qc.h(qr[2])
    qc.h(qr[3])

    # 5) Build syndromes then flag with ONLY simple assignments (QASM3 parser-friendly)
    qc.store(cr[4], expr.bit_xor(cr[0], cr[2]))  # s0 = c0 ^ c2
    qc.store(cr[5], expr.bit_xor(cr[1], cr[3]))  # s1 = c1 ^ c3
    qc.store(cr[6], expr.bit_or(cr[4], cr[5]))   # flag = s0 | s1

    return qc


def conquer_edge_with_d4_n3(client, source_city: str, target_city: str, verbose_qasm_tail: int = 0):
    """
    Finds the specific claimable edge between source_city and target_city,
    submits the D4 N=3 circuit, and prints results + updated status.
    """
    claimable = client.get_claimable_edges()
    if not claimable:
        print("No claimable edges right now (do you own at least one node?).")
        return None

    want = {source_city, target_city}
    target_edge = next((e for e in claimable if set(e["edge_id"]) == want), None)

    if target_edge is None:
        print(f"Edge {source_city} <-> {target_city} is NOT currently claimable from your owned nodes.")
        print("Some claimable edges (first 12):")
        for e in claimable[:12]:
            print(f"  {tuple(e['edge_id'])} | diff={e['difficulty_rating']} | thr={e['base_threshold']}")
        return None

    edge_id = tuple(target_edge["edge_id"])
    diff = int(target_edge.get("difficulty_rating", -1))
    thr  = float(target_edge.get("base_threshold", 0.0))

    print(f"Attacking edge: {edge_id} | difficulty={diff} | threshold={thr:.2f}")
    if diff != 4:
        print("WARNING: This edge is not difficulty 4. This circuit is tuned for D4 (phase-flip-only).")

    circuit = get_d4_opt_circuit_N3()

    # Optional: print tail of QASM3 for debugging
    if verbose_qasm_tail > 0:
        q = qasm3.dumps(circuit).splitlines()
        print("\n=== QASM3 tail preview ===")
        print("\n".join(q[-verbose_qasm_tail:]))
        print("=========================\n")

    # IMPORTANT:
    # - N=3
    # - flag is c[6]
    result = client.claim_edge(
        edge=edge_id,
        circuit=circuit,
        flag_bit=6,
        num_bell_pairs=3
    )

    if not result.get("ok"):
        print("FAILED:", result.get("error", {}).get("code"), "-", result.get("error", {}).get("message"))
        return result

    data = result.get("data", {})
    F = float(data.get("fidelity", 0.0))
    p = float(data.get("success_probability", 0.0))
    print(f"Success: {data.get('success')}")
    print(f"Fidelity: {F:.4f} (threshold: {float(data.get('threshold', 0.0)):.4f})")
    print(f"Success probability: {p:.4f}")
    print(f"Claim strength added (f*p): {F*p:.4f}")

    status = client.get_status()
    print("\n--- Updated Player Status ---")
    print(f"Score: {status.get('score', 0)}")
    print(f"Budget: {status.get('budget', 0)}")
    print(f"Owned nodes: {len(status.get('owned_nodes', []))}")
    print(f"Owned edges: {len(status.get('owned_edges', []))}")
    print("----------------------------\n")

    return result

# =========================
# D5 (pX=pZ=0.16 Bell-diagonal) N=5 conquer helper (QASM3 via Qiskit)
# Phase-parity checks FIRST, then bit-parity checks.
# Postselect on final flag == 0.
# Specify SOURCE + TARGET at bottom.
# =========================

def get_d5_opt_circuit_N5_phase_then_bit():
    """
    N=5 (10 qubits). Pairing outside-in:
      pair0: (q0,q9)   phase sacrificial #2
      pair1: (q1,q8)   phase sacrificial #1
      pair2: (q2,q7)   bit   sacrificial #2
      pair3: (q3,q6)   bit   sacrificial #1
      pair4: (q4,q5)   OUTPUT (kept)  <-- MUST be (q[N-1], q[N]) = (q4,q5)

    Classical bits:
      phase meas:  c0=q1, c1=q0, c2=q8, c3=q9
      bit meas:    c4=q3, c5=q2, c6=q6, c7=q7
      syndromes:   c8=s_p1, c9=s_p2, c10=s_b1, c11=s_b2
      temps:       c12=t_phase, c13=t_bit
      flag:        c14=flag (0=keep)
    """
    qr = QuantumRegister(10, "q")
    cr = ClassicalRegister(15, "c")
    qc = QuantumCircuit(qr, cr)

    # ---------- (A) PHASE checks first ----------
    # Convert phase(Z) <-> bit(X) on ONLY the pairs involved:
    # output (q4,q5) and phase sacrificial pairs (q1,q8), (q0,q9)
    qc.h(qr[4]); qc.h(qr[5])
    qc.h(qr[1]); qc.h(qr[8])
    qc.h(qr[0]); qc.h(qr[9])

    # Bilateral parity checks from output -> phase sacrificial
    qc.cx(qr[4], qr[1])  # Alice
    qc.cx(qr[5], qr[8])  # Bob
    qc.cx(qr[4], qr[0])  # Alice
    qc.cx(qr[5], qr[9])  # Bob

    # Measure phase sacrificial qubits (in computational basis == X-basis pre-H)
    qc.measure(qr[1], cr[0])
    qc.measure(qr[0], cr[1])
    qc.measure(qr[8], cr[2])
    qc.measure(qr[9], cr[3])

    # Restore output basis (so final fidelity is evaluated vs |Phi+>)
    qc.h(qr[4]); qc.h(qr[5])

    # ---------- (B) BIT checks second ----------
    # Bit checks are Z-basis parity checks: no H needed.
    # Use different sacrificial pairs (q3,q6) and (q2,q7)
    qc.cx(qr[4], qr[3])  # Alice
    qc.cx(qr[5], qr[6])  # Bob
    qc.cx(qr[4], qr[2])  # Alice
    qc.cx(qr[5], qr[7])  # Bob

    # Measure bit sacrificial qubits
    qc.measure(qr[3], cr[4])
    qc.measure(qr[2], cr[5])
    qc.measure(qr[6], cr[6])
    qc.measure(qr[7], cr[7])

    # ---------- (C) Classical postselection flag ----------
    # Phase syndromes: s_p1 = c0 ^ c2, s_p2 = c1 ^ c3
    qc.store(cr[8],  expr.bit_xor(cr[0], cr[2]))
    qc.store(cr[9],  expr.bit_xor(cr[1], cr[3]))
    qc.store(cr[12], expr.bit_or(cr[8], cr[9]))      # t_phase

    # Bit syndromes: s_b1 = c4 ^ c6, s_b2 = c5 ^ c7
    qc.store(cr[10], expr.bit_xor(cr[4], cr[6]))
    qc.store(cr[11], expr.bit_xor(cr[5], cr[7]))
    qc.store(cr[13], expr.bit_or(cr[10], cr[11]))    # t_bit

    # Final flag = t_phase | t_bit
    qc.store(cr[14], expr.bit_or(cr[12], cr[13]))    # flag (0=keep)

    return qc


def conquer_edge_with_d5_n5(client, source_city: str, target_city: str, verbose_qasm_tail: int = 0):
    claimable = client.get_claimable_edges()
    if not claimable:
        print("No claimable edges right now (do you own at least one node?).")
        return None

    want = {source_city, target_city}
    target_edge = next((e for e in claimable if set(e["edge_id"]) == want), None)

    if target_edge is None:
        print(f"Edge {source_city} <-> {target_city} is NOT currently claimable from your owned nodes.")
        print("Some claimable edges (first 12):")
        for e in claimable[:12]:
            print(f"  {tuple(e['edge_id'])} | diff={e['difficulty_rating']} | thr={e['base_threshold']}")
        return None

    edge_id = tuple(target_edge["edge_id"])
    diff = int(target_edge.get("difficulty_rating", -1))
    thr  = float(target_edge.get("base_threshold", 0.0))

    print(f"Attacking edge: {edge_id} | difficulty={diff} | threshold={thr:.2f}")
    if diff != 5:
        print("WARNING: This edge is not difficulty 5. This circuit is tuned for D5 (both X and Z noise).")

    circuit = get_d5_opt_circuit_N5_phase_then_bit()

    if verbose_qasm_tail > 0:
        q = qasm3.dumps(circuit).splitlines()
        print("\n=== QASM3 tail preview ===")
        print("\n".join(q[-verbose_qasm_tail:]))
        print("=========================\n")

    result = client.claim_edge(
        edge=edge_id,
        circuit=circuit,
        flag_bit=14,       # <-- c[14] is the flag
        num_bell_pairs=5
    )

    if not result.get("ok"):
        print("FAILED:", result.get("error", {}).get("code"), "-", result.get("error", {}).get("message"))
        return result

    data = result.get("data", {})
    F = float(data.get("fidelity", 0.0))
    p = float(data.get("success_probability", 0.0))

    print(f"Success: {data.get('success')}")
    print(f"Fidelity: {F:.4f} (threshold: {float(data.get('threshold', 0.0)):.4f})")
    print(f"Success probability: {p:.4f}")
    print(f"Claim strength added (f*p): {F*p:.4f}")

    status = client.get_status()
    print("\n--- Updated Player Status ---")
    print(f"Score: {status.get('score', 0)}")
    print(f"Budget: {status.get('budget', 0)}")
    print(f"Owned nodes: {len(status.get('owned_nodes', []))}")
    print(f"Owned edges: {len(status.get('owned_edges', []))}")
    print("----------------------------\n")

    return result

if __name__ == "__main__":
    pass
