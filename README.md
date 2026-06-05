# aacp-langchain

**AACP coordination layer for LangChain multi-agent workflows.**

AACP replaces natural language agent-to-agent instructions with typed,
validated, deterministic packets. LangChain handles agent lifecycle,
memory, and tooling. AACP handles what the agents say to each other.

## Install

```bash
pip install aacp-langchain
```

## Quick start

```python
from aacp_langchain import AACPOrchestrator

orch = AACPOrchestrator(model="gpt-4.1-mini")
result = orch.run_workflow("payroll", period="2026-03")
print(result.summary())
```

## Measured results

Payroll workflow comparison. gpt-4.1-mini. June 2026.

```
                        WITHOUT AACP    WITH AACP
Coordination LLM calls:        4            0
Coordination cost:          $0.0024      $0.0000
Agent cost:                 $0.0035      $0.0035
Total cost:                 $0.0059      $0.0035
Total saving:                             18%
Coordination deterministic:    NO          YES
Schema validated:              NO          YES
```

## Department day comparison (59 coordination hops)

```bash
python3 examples/department_comparison.py --mock
```

```
                        WITHOUT AACP    WITH AACP
Coordination LLM calls:       59            0
Coordination cost:         $0.0031      $0.0000
Total saving:                             18%
Deterministic:                 NO          YES
```

## Without AACP vs With AACP

```
Without aacp-langchain:
  Orchestrator
    ↓ "Please ask the HR agent to retrieve salary records..."
  LLM call  ← costs tokens, varies every run
    ↓ "HR Agent, could you fetch the employee salary data?"
  HR Agent

With aacp-langchain:
  Orchestrator
    ↓ FETCH|HR|return:ORCHESTRATOR|p:1|aacp:1.1|res:emp_salary|period:2026-03
  HR Agent  ✓ validates. $0.00 encoding.
```

## Workflows

```python
# Payroll (5 hops)
result = orch.run_workflow("payroll", period="2026-03")

# IT provisioning / JML onboarding (6 hops)
result = orch.run_workflow(
    "it_provisioning",
    username="j.smith",
    dept="Engineering",
    licences=["M365", "Slack"],
)

# Sales qualification (5 hops)
result = orch.run_workflow(
    "sales_qualification",
    lead_id="L-001",
    lead={"budget_gbp": 75000, "timeline_months": 3},
)
```

## Requirements

- Python 3.10+
- `OPENAI_API_KEY`
- `pip install aacp-langchain`

## Links

- Protocol spec: https://aacp.dev
- Python SDK: https://github.com/MackayAndrew/aacp
- CrewAI integration: https://github.com/MackayAndrew/aacp-crewai
- Community rules (241): https://github.com/MackayAndrew/aacp-community-rules
- IETF Draft: https://datatracker.ietf.org/doc/draft-mackay-aacp/

## Licence

MIT
