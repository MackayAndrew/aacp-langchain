"""
aacp-langchain sales qualification example
5-hop BANT qualification using AACP coordination.

Run:
    cd examples
    export OPENAI_API_KEY=sk-...
    python3 sales_workflow.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aacp_langchain import AACPOrchestrator

DEMO_LEADS = [
    {"id": "L-001", "company": "Apex Systems",
     "budget_gbp": 85000, "timeline_months": 3,
     "need_score": 8, "authority_score": 9, "engaged": True},
    {"id": "L-002", "company": "Delta Consulting",
     "budget_gbp": 28000, "timeline_months": 9,
     "need_score": 4, "authority_score": 8, "engaged": False},
    {"id": "L-003", "company": "CoreTech Solutions",
     "budget_gbp": 120000, "timeline_months": 2,
     "need_score": 9, "authority_score": 10, "engaged": True},
]

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first")
        sys.exit(1)

    orch = AACPOrchestrator(model="gpt-4.1-mini")

    print("\nProcessing 3 leads through BANT qualification...")
    print("All coordination via AACP packets — $0.00 encoding cost\n")

    results = []
    for lead in DEMO_LEADS:
        result = orch.run_workflow(
            "sales_qualification",
            lead_id=lead["id"],
            lead=lead,
        )
        score = result.outputs.get("score", {})
        results.append({
            "company":   lead["company"],
            "score":     score.get("total_score"),
            "qualified": score.get("qualified"),
            "cost":      result.total_cost,
        })

    print("\n" + "="*50)
    print("  QUALIFICATION SUMMARY")
    print("="*50)
    for r in results:
        q = "QUALIFIED" if r["qualified"] else "NOT QUALIFIED"
        print(f"  {r['company']:<25} {r['score']:>3}/100  {q:<15} ${r['cost']:.4f}")

    total = sum(r["cost"] for r in results)
    print(f"\n  Total cost for 3 leads: ${total:.4f}")
    print(f"  Encoding cost:          $0.00 (rule-based encoders)")

if __name__ == "__main__":
    main()
