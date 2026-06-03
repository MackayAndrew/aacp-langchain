"""
BaselineOrchestrator
Standard LangChain multi-agent orchestration WITHOUT AACP.

This is the "before" state -- what most agent frameworks do today.
The orchestrator LLM writes natural language instructions to coordinate
specialist agents. Every coordination hop costs tokens and varies
on every run.

Used by examples/comparison.py to demonstrate what AACP replaces.
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from .agent import AuditAgent, MODEL_COSTS


class CoordinationHop:
    """Records one natural language coordination message and its cost."""

    def __init__(
        self,
        from_agent:   str,
        to_agent:     str,
        nl_message:   str,
        response:     dict,
        tokens_in:    int,
        tokens_out:   int,
        cost_usd:     float,
        latency_ms:   float,
    ):
        self.from_agent  = from_agent
        self.to_agent    = to_agent
        self.nl_message  = nl_message
        self.response    = response
        self.tokens_in   = tokens_in
        self.tokens_out  = tokens_out
        self.cost_usd    = cost_usd
        self.latency_ms  = latency_ms
        self.ts          = datetime.now(timezone.utc).isoformat()


class BaselineResult:
    """Accumulated result of a baseline (no AACP) workflow run."""

    def __init__(self, workflow: str, model: str):
        self.workflow         = workflow
        self.model            = model
        self.agent_hops:      list = []
        self.coord_hops:      list[CoordinationHop] = []
        self.outputs:         dict = {}
        self.success          = True
        self.started          = datetime.now(timezone.utc).isoformat()

    @property
    def coordination_cost(self) -> float:
        return round(sum(h.cost_usd for h in self.coord_hops), 6)

    @property
    def agent_cost(self) -> float:
        return round(sum(h.get("cost_usd", 0) for h in self.agent_hops), 6)

    @property
    def total_cost(self) -> float:
        return round(self.coordination_cost + self.agent_cost, 6)

    @property
    def coordination_tokens(self) -> int:
        return sum(h.tokens_in for h in self.coord_hops)

    @property
    def agent_tokens(self) -> int:
        return sum(h.get("tokens_in", 0) for h in self.agent_hops)

    @property
    def total_latency_ms(self) -> float:
        coord = sum(h.latency_ms for h in self.coord_hops)
        agent = sum(h.get("latency_ms", 0) for h in self.agent_hops)
        return round(coord + agent, 0)


class BaselineOrchestrator:
    """
    Standard LangChain orchestration WITHOUT AACP.
    The orchestrator LLM writes natural language coordination messages.
    Every coordination hop is an LLM call.
    """

    COORD_SYSTEM = """You are an orchestrator coordinating a multi-agent workflow.
Write clear, natural language instructions to specialist agents.
Be specific about what data to retrieve or action to take.
Return only the instruction text, no preamble."""

    def __init__(
        self,
        model:      str = "gpt-4.1-mini",
        api_key:    str = None,
        data_dir:   str = "data",
        output_dir: str = "output",
        verbose:    bool = True,
    ):
        self.model      = model
        self.api_key    = api_key or os.environ.get("OPENAI_API_KEY")
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose    = verbose
        self._cpm       = MODEL_COSTS.get(model, 1.0)
        self._llm       = None
        self.output_dir.mkdir(exist_ok=True)

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                max_tokens=300,
                temperature=0.3,  # slight variance to show non-determinism
            )
        return self._llm

    def _write_coordination_message(self, task_description: str) -> tuple[str, int, int, float]:
        """
        Use LLM to write a natural language coordination instruction.
        This is what happens without AACP -- every coordination message
        costs tokens and varies on every run.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        llm   = self._get_llm()
        start = time.time()

        messages = [
            SystemMessage(content=self.COORD_SYSTEM),
            HumanMessage(content=task_description),
        ]
        response   = llm.invoke(messages)
        latency_ms = (time.time() - start) * 1000

        tokens_in  = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
        tokens_out = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
        cost_usd   = (tokens_in + tokens_out) / 1_000_000 * self._cpm

        return response.content, tokens_in, tokens_out, round(cost_usd, 6)

    def _dispatch(
        self,
        from_agent:       str,
        to_agent,
        task_description: str,
        data:             dict = None,
        preview_fn=None,
    ) -> dict | None:
        """
        Write a natural language coordination message via LLM,
        then dispatch it to the receiving agent.
        """
        # This is the coordination LLM call -- the one AACP eliminates
        nl_msg, ti, to_tok, cost = self._write_coordination_message(
            task_description
        )

        if self.verbose:
            print(f"\n  ┌─ [{from_agent}] → [{to_agent.name}]")
            print(f"  │  NL: \"{nl_msg[:72]}\"")
            print(f"  │  Coord cost: ${cost:.4f} ({ti}in/{to_tok}out)")

        # Now call the agent with the natural language message
        response = to_agent.receive(nl_msg, data or {})

        coord_hop = CoordinationHop(
            from_agent  = from_agent,
            to_agent    = to_agent.name,
            nl_message  = nl_msg,
            response    = response.get("result"),
            tokens_in   = ti,
            tokens_out  = to_tok,
            cost_usd    = cost,
            latency_ms  = (ti + to_tok) / 100,  # approx
        )

        if self.verbose:
            if response.get("error"):
                print(f"  └─ ✗ ERROR: {response['error'][:60]}")
            else:
                print(
                    f"  └─ ✓ agent {response.get('tokens_in',0)}in/"
                    f"{response.get('tokens_out',0)}out"
                    f"  ${response.get('cost_usd',0):.4f}"
                )
                if preview_fn and response.get("result"):
                    print(f"     ↳ {str(preview_fn(response['result']))[:72]}")

        return coord_hop, response

    def run_payroll(self, period: str = "2026-03") -> BaselineResult:
        """
        Same payroll workflow as AACPOrchestrator but WITHOUT AACP.
        Orchestrator LLM writes every coordination message.
        """
        import csv

        from .agents import HRAgent, FinanceAgent
        hr_agent  = HRAgent(model=self.model, api_key=self.api_key)
        fin_agent = FinanceAgent(model=self.model, api_key=self.api_key)
        audit     = AuditAgent()

        result = BaselineResult("payroll", self.model)

        emp_file = self.data_dir / f"employees_{period}.csv"
        bud_file = self.data_dir / f"budgets_{period}.csv"
        emp_data = list(csv.DictReader(open(emp_file))) if emp_file.exists() else []
        bud_data = list(csv.DictReader(open(bud_file))) if bud_file.exists() else []

        print(f"\n{'='*60}")
        print(f"  BASELINE (no AACP) — PAYROLL {period}")
        print(f"  Model: {self.model}")
        print(f"  Coordination: natural language via LLM each hop")
        print(f"{'='*60}")

        # Hop 1 -- orchestrator writes NL instruction to HR agent
        coord1, r1 = self._dispatch(
            "ORCHESTRATOR", hr_agent,
            f"Please retrieve all active employee salary records for the period "
            f"ending {period}. Include employee ID, name, department, cost centre, "
            f"base salary, any delta, and pension contribution rate. Return as JSON.",
            {"employees": emp_data, "period": period},
            lambda x: f"{x.get('total_employees',0)} employees",
        )
        result.coord_hops.append(coord1)
        result.agent_hops.append(r1)
        if not r1.get("result"): result.success = False; return result

        # Hop 2 -- orchestrator writes NL instruction to Finance agent
        coord2, r2 = self._dispatch(
            "ORCHESTRATOR", fin_agent,
            f"Please retrieve the cost centre budget allocations for period {period}. "
            f"Calculate utilisation percentage and flag any cost centres that are "
            f"above 85% of their approved annual budget. Return as JSON.",
            {"budgets": bud_data, "period": period},
            lambda x: f"{x.get('flagged_count',0)} flagged",
        )
        result.coord_hops.append(coord2)
        result.agent_hops.append(r2)
        if not r2.get("result"): result.success = False; return result

        # Hop 3 -- orchestrator writes NL instruction for merge/calculate
        coord3, r3 = self._dispatch(
            "ORCHESTRATOR", hr_agent,
            f"Using the employee salary data and cost centre budget information "
            f"provided, calculate the full payroll for {period}. Apply PAYE at "
            f"20%, calculate pension contributions, and compute net pay for each "
            f"employee. Flag any cost centres where total payroll would breach "
            f"90% of the approved annual budget. Return as JSON.",
            {"employees": r1.get("result"), "budgets": r2.get("result")},
            lambda x: f"£{x.get('totals',{}).get('gross',0):,} gross",
        )
        result.coord_hops.append(coord3)
        result.agent_hops.append(r3)
        if not r3.get("result"): result.success = False; return result

        # Hop 4 -- orchestrator writes NL instruction for report
        coord4, r4 = self._dispatch(
            "ORCHESTRATOR", hr_agent,
            f"Generate an executive payroll summary report for {period} based on "
            f"the calculated payroll data. Include key figures, highlight any "
            f"anomalies or budget breaches, and provide recommended actions. "
            f"Return as JSON.",
            {"payroll_summary": r3.get("result"), "period": period},
            lambda x: str(x.get("executive_summary",""))[:60],
        )
        result.coord_hops.append(coord4)
        result.agent_hops.append(r4)

        # Audit (deterministic, no LLM)
        audit.receive("LOG|HR|return:AUD-Agent|p:2|aacp:1.1|status:complete")

        result.outputs = {
            "employees": r1.get("result"),
            "budgets":   r2.get("result"),
            "payroll":   r3.get("result"),
            "report":    r4.get("result"),
        }

        t = result
        print(f"\n{'─'*60}")
        print(f"  COMPLETE — baseline payroll (no AACP)")
        print(f"  Coordination LLM calls:  {len(result.coord_hops)}")
        print(f"  Coordination cost:       ${t.coordination_cost:.4f}")
        print(f"  Coordination tokens:     {t.coordination_tokens:,}")
        print(f"  Agent cost:              ${t.agent_cost:.4f}")
        print(f"  Total cost:              ${t.total_cost:.4f}")
        print(f"{'─'*60}")

        return result
