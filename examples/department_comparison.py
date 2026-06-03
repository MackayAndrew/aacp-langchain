"""
examples/department_comparison.py
==================================
Full department day comparison: WITHOUT AACP vs WITH AACP.

5 workflows chained together:
  1. JML Onboarding       3 hires  × 6 hops = 18
  2. Payroll                         5 hops =  5
  3. Sales Qualification  3 leads  × 5 hops = 15
  4. CS Resolution        3 tickets× 5 hops = 15
  5. Month-End Close                 6 hops =  6
  ─────────────────────────────────────────────
  Total coordination hops:                   59

Run:
    python3 examples/department_comparison.py --mock
    python3 examples/department_comparison.py --data ../aacp-lab/data
"""

import sys, os, json, argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aacp_langchain.workflows.department_day import (
    run_department_day, write_mock_data
)
from aacp_langchain.packet_bus    import AACPPacketBus
from aacp_langchain.agent         import AuditAgent
from aacp_langchain.agents        import (
    HRAgent, FinanceAgent, ITAgent, SalesAgent, CSAgent
)
from aacp_langchain.baseline_orchestrator import BaselineOrchestrator


# ── Baseline version (without AACP) ───────────────────────────────────────

class BaselineDepartmentDay:
    """
    Same 5 workflows but orchestrator writes natural language
    coordination for every hop. Tracks coordination LLM calls.
    """

    COORD_PROMPTS = {
        # JML
        "jml_fetch":    "Retrieve the new hire employee record for {arg} from the HR system including role, department and required systems access.",
        "jml_account":  "Create an Active Directory and Entra ID user account for {arg}. Set up email, group memberships, and temporary password.",
        "jml_licences": "Assign the required software licences to {arg}: {extra}. Confirm assignment and any failed licences.",
        "jml_access":   "Configure full system access profile for {arg} including VPN, SharePoint and all role-appropriate systems.",
        "jml_welcome":  "Send a welcome email to {arg} with their credentials, first day instructions, and IT contact details.",
        "jml_log":      "Write the provisioning completion audit record for {arg} to the IT compliance trail.",
        # Payroll
        "pay_emp":      "Retrieve all active employee salary records for period {arg} including salary, cost centre and pension rate. Return as JSON.",
        "pay_budget":   "Retrieve cost centre budget data for {arg}. Calculate YTD utilisation and flag any approaching or breaching their annual budget.",
        "pay_merge":    "Using the employee and budget data provided, calculate the full payroll for {arg}. Apply PAYE at 20%, pension deductions, and flag budget breaches.",
        "pay_report":   "Generate an executive payroll summary report for {arg} highlighting anomalies, budget breaches and recommended actions.",
        "pay_log":      "Write the payroll run audit record for period {arg} to the HR compliance trail.",
        # Sales
        "sal_fetch":    "Retrieve the full lead profile for {arg} from the CRM including engagement history, budget and decision-making authority.",
        "sal_score":    "Score lead {arg} against BANT qualification criteria. Assign scores for Budget, Authority, Need and Timeline. Determine if qualified.",
        "sal_route":    "Based on the BANT score for {arg}, route this lead to the appropriate sales rep or nurture sequence with recommended next steps.",
        "sal_log":      "Write the lead qualification outcome for {arg} to the CRM audit trail.",
        "sal_notify":   "Send a notification to the assigned sales rep about qualified lead {arg} with context and recommended next steps.",
        # CS
        "cs_fetch":     "Retrieve the customer profile for {arg} including lifetime value, loyalty years, complaint history and current sentiment.",
        "cs_triage":    "Triage the complaint in ticket {arg}. Categorise by intent, score sentiment and priority, and determine if escalation is required.",
        "cs_resolve":   "Generate a resolution strategy for ticket {arg} considering the customer LTV and loyalty. Include tone guidance and goodwill recommendation if appropriate.",
        "cs_send":      "Send the resolution response to the customer for ticket {arg} via their preferred channel.",
        "cs_log":       "Write the ticket resolution outcome for {arg} to the CS audit trail.",
        # Month-end
        "me_balance":   "Retrieve the trial balance and open items from the GL for period {arg} for month-end close processing.",
        "me_recon":     "Perform bank reconciliation for period {arg}. Match GL transactions against bank statements and report unmatched items.",
        "me_accruals":  "Calculate and post period accruals for {arg} to the GL in accordance with the accrual policy. Report all journal entries made.",
        "me_variance":  "Run variance analysis comparing {arg} actuals against prior period and budget. Flag material variances for CFO attention.",
        "me_accounts":  "Generate the management accounts pack for {arg} for CFO and Finance Director review.",
        "me_certify":   "Write the month-end close certification record for {arg} to the finance audit trail.",
    }

    def __init__(self, model, api_key, data_dir, output_dir, verbose=True):
        self.model      = model
        self.api_key    = api_key
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose    = verbose
        from aacp_langchain.baseline_orchestrator import MODEL_COSTS
        self._cpm       = MODEL_COSTS.get(model, 1.0)
        self._llm       = None
        self.coord_hops = []
        self.agent_hops = []

    def _get_llm(self):
        if not self._llm:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model=self.model, api_key=self.api_key,
                                   max_tokens=200, temperature=0.3)
        return self._llm

    def _coord(self, prompt_key, arg="", extra=""):
        """Write one NL coordination message. Counts as one LLM call."""
        import time
        from langchain_core.messages import SystemMessage, HumanMessage
        prompt = self.COORD_PROMPTS[prompt_key].format(arg=arg, extra=extra)
        llm    = self._get_llm()
        start  = time.time()
        resp   = llm.invoke([
            SystemMessage(content="You are an orchestrator. Write a clear coordination instruction."),
            HumanMessage(content=prompt)
        ])
        ti  = resp.response_metadata.get("token_usage",{}).get("prompt_tokens",0)
        to  = resp.response_metadata.get("token_usage",{}).get("completion_tokens",0)
        cst = (ti + to) / 1_000_000 * self._cpm
        self.coord_hops.append({"message": resp.content, "tokens_in": ti,
                                 "cost_usd": cst})
        if self.verbose:
            print(f"  [NL coord] \"{ resp.content[:65]}\"  ${cst:.4f}")
        return resp.content

    def _agent(self, agent, nl_msg, data=None):
        """Send NL message to agent. Track agent cost."""
        import time
        r = agent.receive(nl_msg, data or {})
        self.agent_hops.append(r)
        if self.verbose and r.get("result"):
            pass  # already printed by coord
        return r.get("result")

    @property
    def coordination_cost(self):
        return round(sum(h["cost_usd"] for h in self.coord_hops), 6)

    @property
    def agent_cost(self):
        return round(sum(h.get("cost_usd",0) for h in self.agent_hops), 6)

    @property
    def total_cost(self):
        return round(self.coordination_cost + self.agent_cost, 6)

    @property
    def coordination_tokens(self):
        return sum(h["tokens_in"] for h in self.coord_hops)

    def run(self, period="2026-03"):
        import csv as _csv, json as _json
        kwargs = {"model": self.model, "api_key": self.api_key}
        agents = {
            "hr":      HRAgent(**kwargs),
            "finance": FinanceAgent(**kwargs),
            "it":      ITAgent(**kwargs),
            "sales":   SalesAgent(**kwargs),
            "cs":      CSAgent(**kwargs),
            "audit":   AuditAgent(),
        }

        emp_data  = list(_csv.DictReader(open(self.data_dir/"employees_2026-03.csv")))
        bud_data  = list(_csv.DictReader(open(self.data_dir/"budgets_2026-03.csv")))
        hire_data = list(_csv.DictReader(open(self.data_dir/"new_hires_2026-03.csv")))
        lead_data = list(_csv.DictReader(open(self.data_dir/"leads_2026-03.csv")))
        tick_data = list(_csv.DictReader(open(self.data_dir/"tickets_2026-03.csv")))
        rules     = _json.load(open(self.data_dir/"payroll_rules.json"))

        print(f"\n  --- Workflow 1/5: JML Onboarding (3 hires × 6 hops = 18) ---")
        for hire in hire_data:
            uid = hire["username"]
            msg1 = self._coord("jml_fetch", uid)
            r1   = self._agent(agents["hr"], msg1, {"employee": hire})
            msg2 = self._coord("jml_account", uid)
            r2   = self._agent(agents["it"],  msg2, {"username": uid, "dept": hire["dept"]})
            msg3 = self._coord("jml_licences", uid, hire.get("licences","M365"))
            r3   = self._agent(agents["it"],  msg3, {"username": uid})
            msg4 = self._coord("jml_access", uid)
            r4   = self._agent(agents["it"],  msg4, {"username": uid})
            msg5 = self._coord("jml_welcome", uid)
            r5   = self._agent(agents["it"],  msg5, {"username": uid})
            msg6 = self._coord("jml_log", uid)
            agents["audit"].receive(msg6, {"username": uid})

        print(f"\n  --- Workflow 2/5: Payroll (5 hops) ---")
        msg = self._coord("pay_emp",    period)
        r1  = self._agent(agents["hr"],      msg, {"employees": emp_data})
        msg = self._coord("pay_budget", period)
        r2  = self._agent(agents["finance"], msg, {"budgets": bud_data})
        msg = self._coord("pay_merge",  period)
        r3  = self._agent(agents["hr"],      msg, {"employees": r1, "budgets": r2})
        msg = self._coord("pay_report", period)
        r4  = self._agent(agents["hr"],      msg, {"payroll": r3})
        msg = self._coord("pay_log",    period)
        agents["audit"].receive(msg, {"period": period})

        print(f"\n  --- Workflow 3/5: Sales Qualification (3 leads × 5 hops = 15) ---")
        for lead in lead_data:
            lid = lead["id"]
            msg = self._coord("sal_fetch",  lid)
            r1  = self._agent(agents["sales"], msg, {"lead": lead})
            msg = self._coord("sal_score",  lid)
            r2  = self._agent(agents["sales"], msg, {"lead": r1})
            msg = self._coord("sal_route",  lid)
            r3  = self._agent(agents["sales"], msg, {"score": r2})
            msg = self._coord("sal_log",    lid)
            agents["audit"].receive(msg, {"lead_id": lid})
            msg = self._coord("sal_notify", lid)
            r5  = self._agent(agents["sales"], msg, {"routing": r3})

        print(f"\n  --- Workflow 4/5: CS Resolution (3 tickets × 5 hops = 15) ---")
        for ticket in tick_data:
            tid = ticket["id"]
            msg = self._coord("cs_fetch",   ticket["customer_id"])
            r1  = self._agent(agents["cs"], msg, {"ticket": ticket})
            msg = self._coord("cs_triage",  tid)
            r2  = self._agent(agents["cs"], msg, {"ticket": ticket, "customer": r1})
            msg = self._coord("cs_resolve", tid)
            r3  = self._agent(agents["cs"], msg, {"triage": r2})
            msg = self._coord("cs_send",    tid)
            r4  = self._agent(agents["cs"], msg, {"resolution": r3})
            msg = self._coord("cs_log",     tid)
            agents["audit"].receive(msg, {"ticket_id": tid})

        print(f"\n  --- Workflow 5/5: Month-End Close (6 hops) ---")
        for key, arg in [("me_balance",period),("me_recon",period),
                         ("me_accruals",period),("me_variance",period),
                         ("me_accounts",period),("me_certify",period)]:
            msg = self._coord(key, arg)
            if "certify" in key:
                agents["audit"].receive(msg, {"period": period})
            else:
                self._agent(agents["finance"], msg, {"period": period})


# ── AACP version ───────────────────────────────────────────────────────────

def run_aacp_department_day(model, api_key, data_dir, output_dir, period, verbose):
    kwargs = {"model": model, "api_key": api_key}
    agents = {
        "hr":      HRAgent(**kwargs),
        "finance": FinanceAgent(**kwargs),
        "it":      ITAgent(**kwargs),
        "sales":   SalesAgent(**kwargs),
        "cs":      CSAgent(**kwargs),
        "audit":   AuditAgent(),
    }
    bus = AACPPacketBus(
        workflow="department_day", model=model,
        audit_log=f"{output_dir}/audit_aacp.jsonl",
        verbose=verbose,
    )
    run_department_day(bus, agents, Path(data_dir), period)
    return bus.result


# ── Comparison table ───────────────────────────────────────────────────────

def print_comparison(b, a, model):
    w = 64
    print(f"\n{'='*w}")
    print(f"  DEPARTMENT DAY COMPARISON")
    print(f"  Model: {model}  |  Period: 2026-03")
    print(f"  5 workflows  |  ~59 coordination hops")
    print(f"{'='*w}")
    print(f"  {'Metric':<38} {'NO AACP':>10} {'AACP':>10}")
    print(f"  {'─'*60}")

    def row(label, bv, av, hi=False):
        mark = " ←" if hi else ""
        print(f"  {label:<38} {str(bv):>10} {str(av):>10}{mark}")

    b_coord = len(b.coord_hops)
    a_coord = 0

    row("Coordination approach",       "LLM text",   "AACP packet")
    row("Coordination LLM calls",       b_coord,      a_coord,      hi=True)
    row("Coordination cost (USD)",
        f"${b.coordination_cost:.4f}", "$0.0000",                   hi=True)
    row("Coordination tokens (approx)", b.coordination_tokens,
        sum(max(1,len(h.packet)//4) for h in a.hops))
    print(f"  {'─'*60}")
    row("Agent LLM calls",
        len(b.agent_hops), len(a.hops))
    row("Agent cost (USD)",
        f"${b.agent_cost:.4f}", f"${a.total_cost:.4f}")
    print(f"  {'─'*60}")
    row("Total cost (USD)",
        f"${b.total_cost:.4f}", f"${a.total_cost:.4f}",            hi=True)
    saving = b.total_cost - a.total_cost
    saving_pct = (saving / b.total_cost * 100) if b.total_cost > 0 else 0
    row("Total saving",
        "",  f"${saving:.4f} ({saving_pct:.0f}%)",                 hi=True)
    print(f"  {'─'*60}")
    row("Coordination deterministic",  "NO",  "YES",               hi=True)
    row("Schema validated",            "NO",  "YES",               hi=True)
    row("Audit trail structured",      "NO",  "YES",               hi=True)
    row("Same output every run",       "NO",  "YES",               hi=True)
    print(f"  {'='*60}")

    print(f"\n  SUMMARY")
    print(f"  Coordination LLM calls eliminated: {b_coord}")
    print(f"  Coordination cost saved:            ${b.coordination_cost:.4f}")
    print(f"  Total cost reduction:               {saving_pct:.0f}%")
    print(f"  Deterministic coordination:         YES")
    print(f"  Every packet schema-validated:      YES")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gpt-4.1-mini")
    parser.add_argument("--data",   default="data")
    parser.add_argument("--mock",   action="store_true")
    parser.add_argument("--output", default="output")
    parser.add_argument("--verbose",action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)

    data_dir = Path(args.data)
    if args.mock:
        print("Writing mock data...")
        write_mock_data(data_dir)

    if not (data_dir / "employees_2026-03.csv").exists():
        print(f"ERROR: data files not found in {data_dir}")
        print("Run with --mock or point --data to aacp-lab/data")
        sys.exit(1)

    Path(args.output).mkdir(exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  AACP-LangChain Department Day Comparison")
    print(f"  5 workflows  |  ~59 coordination hops each")
    print(f"  Model: {args.model}")
    print(f"{'='*64}")

    # Run WITHOUT AACP
    print(f"\nRUN 1: WITHOUT AACP (natural language coordination)")
    print(f"Each coordination hop is an LLM call.")
    baseline = BaselineDepartmentDay(
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        data_dir=str(data_dir),
        output_dir=args.output,
        verbose=args.verbose,
    )
    baseline.run(period="2026-03")

    print(f"\nRUN 2: WITH AACP (rule-based packet coordination)")
    print(f"Coordination encoding cost: $0.00")
    aacp_result = run_aacp_department_day(
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        data_dir=str(data_dir),
        output_dir=args.output,
        period="2026-03",
        verbose=args.verbose,
    )

    print_comparison(baseline, aacp_result, args.model)

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "workflows": 5,
        "coordination_hops": 59,
        "baseline": {
            "coordination_llm_calls": len(baseline.coord_hops),
            "coordination_cost_usd":  baseline.coordination_cost,
            "coordination_tokens":    baseline.coordination_tokens,
            "agent_cost_usd":         baseline.agent_cost,
            "total_cost_usd":         baseline.total_cost,
            "deterministic": False, "validated": False,
        },
        "with_aacp": {
            "coordination_llm_calls": 0,
            "coordination_cost_usd":  0.0,
            "agent_cost_usd":         aacp_result.total_cost,
            "total_cost_usd":         aacp_result.total_cost,
            "deterministic": True, "validated": True,
        },
    }
    p = f"{args.output}/dept_comparison_{ts}.json"
    with open(p,"w") as f: json.dump(out,f,indent=2)
    print(f"  Saved: {p}\n")


if __name__ == "__main__":
    main()
