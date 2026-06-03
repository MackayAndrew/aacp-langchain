"""
aacp-langchain payroll workflow example
Full 5-hop payroll run using AACP coordination.

Requires data files in ../data/:
  employees_2026-03.csv
  budgets_2026-03.csv
  payroll_rules.json

These are available in the aacp-lab repo:
  github.com/MackayAndrew/aacp-lab

Run:
    cd examples
    export OPENAI_API_KEY=sk-...
    python3 payroll_workflow.py [--model gpt-4.1]
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aacp_langchain import AACPOrchestrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gpt-4.1-mini")
    parser.add_argument("--period", default="2026-03")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first")
        sys.exit(1)

    orch = AACPOrchestrator(
        model=args.model,
        data_dir="../data",
    )

    result = orch.run_workflow("payroll", period=args.period)
    print("\n" + result.summary())

    # Show what was audited
    import json
    from pathlib import Path
    audit = Path("output/audit.jsonl")
    if audit.exists():
        lines = audit.read_text().strip().split("\n")
        print(f"\nAudit trail: {len(lines)} records written to output/audit.jsonl")

if __name__ == "__main__":
    main()
