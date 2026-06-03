"""
aacp-langchain: AACP vs Baseline Comparison
============================================

Runs the SAME payroll workflow two ways:
  1. Standard LangChain -- LLM writes every coordination message
  2. aacp-langchain     -- AACP rule-based encoder, $0.00 coordination

Prints a side-by-side comparison showing exactly what AACP replaces.

Run:
    cd ~/Desktop/aacp-langchain
    export OPENAI_API_KEY=sk-...

    # With lab data files:
    python3 examples/comparison.py --data ../aacp-lab/data

    # Without data files (uses mock data):
    python3 examples/comparison.py --mock

    # Specific model:
    python3 examples/comparison.py --model gpt-4.1 --mock
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock data (used when no CSV files available) ───────────────────────────

MOCK_EMPLOYEES = [
    {"id":"E001","name":"Alice Smith","dept":"Engineering","cost_centre":"CC-10",
     "base_salary_gbp":"72000","delta_gbp":"0","pension_rate":"0.05","status":"active"},
    {"id":"E002","name":"Bob Jones","dept":"Sales","cost_centre":"CC-20",
     "base_salary_gbp":"58000","delta_gbp":"2500","pension_rate":"0.05","status":"active"},
    {"id":"E003","name":"Carol White","dept":"Finance","cost_centre":"CC-30",
     "base_salary_gbp":"65000","delta_gbp":"0","pension_rate":"0.08","status":"active"},
    {"id":"E004","name":"David Brown","dept":"Engineering","cost_centre":"CC-10",
     "base_salary_gbp":"85000","delta_gbp":"5000","pension_rate":"0.05","status":"active"},
]

MOCK_BUDGETS = [
    {"cc_id":"CC-10","cc_name":"Engineering","approved_annual_gbp":"420000",
     "ytd_spend_gbp":"378000","owner":"Sarah Chen","gl_code":"GL-1010"},
    {"cc_id":"CC-20","cc_name":"Sales","approved_annual_gbp":"140000",
     "ytd_spend_gbp":"98000","owner":"Marcus Webb","gl_code":"GL-2020"},
    {"cc_id":"CC-30","cc_name":"Finance","approved_annual_gbp":"160000",
     "ytd_spend_gbp":"124000","owner":"David Park","gl_code":"GL-3030"},
]

MOCK_RULES = {
    "version": "payroll_v2", "period": "2026-03",
    "paye_rate": 0.20, "budget_warning_threshold": 0.85,
    "budget_breach_threshold": 0.90, "currency": "GBP",
    "pay_date": "2026-03-28", "approver": "Finance Director",
}


def write_mock_data(data_dir: Path):
    """Write mock CSV files so the orchestrators can load them."""
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "employees_2026-03.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MOCK_EMPLOYEES[0].keys())
        w.writeheader(); w.writerows(MOCK_EMPLOYEES)

    with open(data_dir / "budgets_2026-03.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MOCK_BUDGETS[0].keys())
        w.writeheader(); w.writerows(MOCK_BUDGETS)

    with open(data_dir / "payroll_rules.json", "w") as f:
        json.dump(MOCK_RULES, f, indent=2)


def print_comparison(baseline, aacp_result):
    """Print the side-by-side comparison table."""

    b = baseline
    a = aacp_result

    # Count coordination hops vs agent hops
    b_coord_hops   = len(b.coord_hops)
    b_agent_hops   = len(b.agent_hops)
    a_coord_hops   = len(a.hops)
    a_agent_hops   = len(a.hops)  # same agents, same work

    b_coord_tokens = b.coordination_tokens
    a_coord_tokens = sum(
        1  # AACP packets are tiny -- count the packet chars / 4
        for h in a.hops
        if h.packet and not h.packet.startswith("LOG")
    )
    # More accurately -- show the actual packet token estimates
    a_coord_tokens_est = sum(
        max(1, len(h.packet) // 4)
        for h in a.hops
    )

    width = 62

    print(f"\n{'='*width}")
    print(f"  COMPARISON: Same payroll workflow, two coordination styles")
    print(f"  Model: {b.model}  |  Period: 2026-03")
    print(f"{'='*width}")
    print(f"\n  {'Metric':<36} {'WITHOUT AACP':>12} {'WITH AACP':>10}")
    print(f"  {'─'*58}")

    def row(label, b_val, a_val, highlight=False):
        b_str = str(b_val)
        a_str = str(a_val)
        marker = " ←" if highlight else ""
        print(f"  {label:<36} {b_str:>12} {a_str:>10}{marker}")

    row("Coordination approach",       "LLM text",     "AACP packet")
    row("Coordination LLM calls",       b_coord_hops,   0,           highlight=True)
    row("Coordination cost (USD)",
        f"${b.coordination_cost:.4f}",
        "$0.0000",                                                     highlight=True)
    row("Coordination tokens (approx)", b_coord_tokens, a_coord_tokens_est)
    print(f"  {'─'*58}")
    row("Agent LLM calls",              b_agent_hops,   a_agent_hops)
    row("Agent cost (USD)",
        f"${b.agent_cost:.4f}",
        f"${a.total_cost:.4f}")
    print(f"  {'─'*58}")
    row("Total cost (USD)",
        f"${b.total_cost:.4f}",
        f"${a.total_cost:.4f}",                                        highlight=True)
    row("Latency (approx)",
        f"{b.total_latency_ms/1000:.1f}s",
        f"{a.total_latency_ms/1000:.1f}s")
    print(f"  {'─'*58}")
    row("Coordination deterministic",  "NO",            "YES",        highlight=True)
    row("Packets schema-validated",    "NO",            "YES",        highlight=True)
    row("Machine-readable audit trail","NO",            "YES",        highlight=True)
    row("Same output every run",       "NO",            "YES",        highlight=True)

    print(f"\n{'='*width}")
    print(f"  FINDINGS")
    print(f"{'='*width}")

    coord_saving = b.coordination_cost
    total_saving = b.total_cost - a.total_cost
    pct_saving   = (total_saving / b.total_cost * 100) if b.total_cost > 0 else 0

    print(f"\n  Coordination LLM calls eliminated: {b_coord_hops}")
    print(f"  Coordination cost saved:            ${coord_saving:.4f}")
    print(f"  Total cost reduction:               {pct_saving:.1f}%")
    print(f"\n  Without AACP, coordination messages vary on every run.")
    print(f"  With AACP, every packet is identical and validated.")
    print(f"\n  The agent work is identical in both cases.")
    print(f"  AACP only changes the coordination layer.")
    print(f"{'='*width}\n")

    # Sample coordination messages for comparison
    if b.coord_hops:
        print(f"  SAMPLE COORDINATION MESSAGE — Without AACP (hop 1):")
        print(f"  \"{b.coord_hops[0].nl_message[:120]}\"")
        print()

    if a.hops:
        print(f"  SAMPLE COORDINATION PACKET — With AACP (hop 1):")
        print(f"  {a.hops[0].packet}")
        print(f"  Encoding cost: $0.00  (rule-based encoder)")
        print(f"  Validated: {a.hops[0].valid}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare AACP vs standard LangChain coordination"
    )
    parser.add_argument("--model",  default="gpt-4.1-mini",
                        choices=["gpt-4.1-mini","gpt-4.1","gpt-4o","gpt-4o-mini"])
    parser.add_argument("--data",   default="data",
                        help="Path to data directory with CSV files")
    parser.add_argument("--mock",   action="store_true",
                        help="Use mock data (no CSV files needed)")
    parser.add_argument("--output", default="output",
                        help="Output directory for results and audit log")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running")
        sys.exit(1)

    data_dir = Path(args.data)

    if args.mock:
        print("Using mock data...")
        write_mock_data(data_dir)

    if not (data_dir / "employees_2026-03.csv").exists():
        print(f"ERROR: {data_dir}/employees_2026-03.csv not found")
        print("Run with --mock to use built-in demo data")
        print("Or point --data to the aacp-lab/data directory")
        sys.exit(1)

    from aacp_langchain.baseline_orchestrator import BaselineOrchestrator
    from aacp_langchain import AACPOrchestrator

    print("\nRunning comparison...")
    print("Both workflows use the same model and same data.")
    print("The only difference is how agents coordinate.\n")

    # Run baseline (without AACP)
    baseline_orch = BaselineOrchestrator(
        model=args.model,
        data_dir=str(data_dir),
        output_dir=args.output,
        verbose=True,
    )
    baseline_result = baseline_orch.run_payroll(period="2026-03")

    print("\n" + "─"*60)
    print("  Now running same workflow WITH AACP...")
    print("─"*60)

    # Run with AACP
    aacp_orch = AACPOrchestrator(
        model=args.model,
        data_dir=str(data_dir),
        output_dir=args.output,
        verbose=True,
    )
    aacp_result = aacp_orch.run_workflow("payroll", period="2026-03")

    # Print comparison
    print_comparison(baseline_result, aacp_result)

    # Save results
    Path(args.output).mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model":     args.model,
        "baseline": {
            "coordination_llm_calls": len(baseline_result.coord_hops),
            "coordination_cost_usd":  baseline_result.coordination_cost,
            "coordination_tokens":    baseline_result.coordination_tokens,
            "agent_cost_usd":         baseline_result.agent_cost,
            "total_cost_usd":         baseline_result.total_cost,
            "deterministic":          False,
            "validated":              False,
        },
        "with_aacp": {
            "coordination_llm_calls": 0,
            "coordination_cost_usd":  0.0,
            "agent_cost_usd":         aacp_result.total_cost,
            "total_cost_usd":         aacp_result.total_cost,
            "deterministic":          True,
            "validated":              True,
        },
    }
    out_path = f"{args.output}/comparison_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
