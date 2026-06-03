"""
aacp-langchain quickstart
Simplest possible working example.

Run:
    cd examples
    export OPENAI_API_KEY=sk-...
    python3 quickstart.py

What this demonstrates:
  - AACP packets coordinating LangChain agents
  - Rule-based encoder: $0.00 packet generation
  - LLM calls only in specialist agents (the actual work)
  - Full audit trail written automatically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aacp_langchain import AACPOrchestrator

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first")
        sys.exit(1)

    print("\naacp-langchain quickstart")
    print("="*50)
    print("AACP coordinates agents with typed packets.")
    print("LLM calls happen in agents, not coordination.\n")

    # Use cheapest model for quickstart
    orch = AACPOrchestrator(model="gpt-4.1-mini")

    # Run IT provisioning -- fastest workflow (6 hops)
    result = orch.run_workflow(
        "it_provisioning",
        username="j.smith",
        dept="Engineering",
        licences=["M365", "Slack"],
    )

    print("\n" + result.summary())
    print(f"\nEncoding cost: $0.00 (rule-based encoder)")
    print(f"Agent cost:    ${result.total_cost:.4f}")
    print(f"Total hops:    {len(result.hops)}")

if __name__ == "__main__":
    main()
