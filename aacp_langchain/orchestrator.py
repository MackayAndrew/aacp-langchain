"""
AACPOrchestrator
High-level interface for running AACP-coordinated workflows
inside LangChain.

The orchestrator:
  1. Loads the right workflow module
  2. Instantiates agents with the specified model
  3. Builds AACP packets using rule-based encoders ($0.00)
  4. Dispatches packets via AACPPacketBus
  5. Returns a WorkflowResult with full audit trail

Usage:
    from aacp_langchain import AACPOrchestrator

    orch = AACPOrchestrator(model="gpt-4.1-mini")
    result = orch.run_workflow("payroll", period="2026-03")
    print(result.summary())
"""

import os
from pathlib import Path
from .packet_bus import AACPPacketBus, WorkflowResult
from .agent      import AuditAgent
from .agents     import HRAgent, FinanceAgent, ITAgent, SalesAgent, CSAgent

SUPPORTED_WORKFLOWS = ["payroll", "it_provisioning", "sales_qualification"]


class AACPOrchestrator:
    """
    Coordinates a multi-agent LangChain workflow using AACP packets.
    The orchestrator never uses LLM calls for coordination --
    only rule-based encoders. LLM calls happen only in the
    specialist agents doing the actual work.
    """

    def __init__(
        self,
        model:     str = "gpt-4.1-mini",
        api_key:   str = None,
        data_dir:  str = "data",
        output_dir: str = "output",
        verbose:   bool = True,
    ):
        self.model      = model
        self.api_key    = api_key or os.environ.get("OPENAI_API_KEY")
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose    = verbose

        # Initialise agents
        kwargs = {"model": self.model, "api_key": self.api_key}
        self.hr_agent      = HRAgent(**kwargs)
        self.finance_agent = FinanceAgent(**kwargs)
        self.it_agent      = ITAgent(**kwargs)
        self.sales_agent   = SalesAgent(**kwargs)
        self.cs_agent      = CSAgent(**kwargs)
        self.audit_agent   = AuditAgent()

        self.output_dir.mkdir(exist_ok=True)

    def run_workflow(self, workflow: str, **kwargs) -> WorkflowResult:
        """
        Run a named workflow. Returns WorkflowResult with full audit trail.

        Supported workflows:
            payroll            kwargs: period="2026-03"
            it_provisioning    kwargs: username="j.smith", dept="Engineering"
            sales_qualification kwargs: lead_id="L-001"
        """
        print(f"\n{'='*60}")
        print(f"  AACP-LangChain: {workflow.upper()}")
        print(f"  Model: {self.model}")
        print(f"{'='*60}")

        bus = AACPPacketBus(
            workflow=workflow,
            model=self.model,
            audit_log=str(self.output_dir / "audit.jsonl"),
            verbose=self.verbose,
        )

        if workflow == "payroll":
            self._run_payroll(bus, **kwargs)
        elif workflow == "it_provisioning":
            self._run_it_provisioning(bus, **kwargs)
        elif workflow == "sales_qualification":
            self._run_sales_qualification(bus, **kwargs)
        else:
            raise ValueError(
                f"Unknown workflow: {workflow}. "
                f"Supported: {SUPPORTED_WORKFLOWS}"
            )

        t = bus.result
        print(f"\n{'─'*60}")
        print(f"  COMPLETE — {workflow}")
        print(f"  Hops:    {len(t.hops)}")
        print(f"  Tokens:  {t.total_tokens:,}")
        print(f"  Cost:    ${t.total_cost:.4f}")
        print(f"  Time:    {t.total_latency_ms/1000:.1f}s")
        print(f"{'─'*60}")

        return bus.result

    # ── Payroll workflow ──────────────────────────────────────────────────────

    def _run_payroll(self, bus: AACPPacketBus, period: str = "2026-03"):
        """
        5-hop payroll workflow using AACP packets throughout.
        Uses PayrollEncoder for zero-cost packet generation.
        """
        import csv, json as _json

        try:
            from aacp.encoders.workflows.payroll import PayrollEncoder
            enc = PayrollEncoder()
        except ImportError:
            raise ImportError("pip install aacp to use payroll workflow")

        # Load data
        emp_file = self.data_dir / f"employees_{period}.csv"
        bud_file = self.data_dir / f"budgets_{period}.csv"
        rul_file = self.data_dir / "payroll_rules.json"

        emp_data = list(csv.DictReader(open(emp_file))) if emp_file.exists() else []
        bud_data = list(csv.DictReader(open(bud_file))) if bud_file.exists() else []
        rules    = _json.load(open(rul_file)) if rul_file.exists() else {}

        # Hop 1: Fetch employees — AACP packet, $0.00 encoding
        pkt1 = enc.fetch_employees(period)
        r1 = bus.dispatch(
            "ORCHESTRATOR", self.hr_agent, pkt1.packet,
            {"employees": emp_data, "period": period},
            lambda x: f"{x.get('total_employees',0)} employees, gross £{x.get('total_gross',0):,}",
        )
        if not r1: return

        # Hop 2: Fetch budgets — AACP packet, $0.00 encoding
        pkt2 = enc.fetch_budgets(period)
        r2 = bus.dispatch(
            "ORCHESTRATOR", self.finance_agent, pkt2.packet,
            {"budgets": bud_data, "period": period},
            lambda x: f"{x.get('flagged_count',0)} CCs pre-flagged",
        )
        if not r2: return

        # Hop 3: Merge and calculate — AACP packet, $0.00 encoding
        pkt3 = enc.merge_and_calculate(period)
        r3 = bus.dispatch(
            "ORCHESTRATOR", self.hr_agent, pkt3.packet,
            {"employees": r1, "budgets": r2, "rules": rules},
            lambda x: f"£{x.get('totals',{}).get('gross',0):,} gross | {len(x.get('anomalies',[]))} anomalies",
        )
        if not r3: return

        # Hop 4: Generate report — AACP packet, $0.00 encoding
        pkt4 = enc.generate_report(period, period)
        r4 = bus.dispatch(
            "ORCHESTRATOR", self.hr_agent, pkt4.packet,
            {"payroll_summary": r3, "period": period, "rules": rules},
            lambda x: str(x.get("executive_summary",""))[:70],
        )

        # Hop 5: Audit — deterministic, $0.00
        pkt5 = enc.log_run(period, status="review_required")
        bus.dispatch(
            "ORCHESTRATOR", self.audit_agent, pkt5.packet,
            {"period": period, "workflow": "payroll"},
            lambda x: "Audit record written",
        )

        bus.result.outputs = {"employees": r1, "budgets": r2,
                              "payroll": r3, "report": r4}

    # ── IT provisioning workflow ──────────────────────────────────────────────

    def _run_it_provisioning(
        self,
        bus:      AACPPacketBus,
        username: str = "j.smith",
        dept:     str = "Engineering",
        licences: list = None,
    ):
        """6-hop JML onboarding workflow using AACP packets."""
        try:
            from aacp.encoders.workflows.jml import JMLEncoder
            enc = JMLEncoder()
        except ImportError:
            raise ImportError("pip install aacp to use IT provisioning workflow")

        licences = licences or ["M365", "Slack", "VPN"]

        # Hop 1: Fetch employee record
        pkt1 = enc.fetch_new_hire(f"E-{username}")
        r1 = bus.dispatch(
            "ORCHESTRATOR", self.hr_agent, pkt1.packet,
            {"username": username, "dept": dept},
            lambda x: f"{x.get('employee',{}).get('name', username)} — {dept}",
        )

        # Hop 2: Create AD account
        pkt2 = enc.create_account(username, dept)
        r2 = bus.dispatch(
            "ORCHESTRATOR", self.it_agent, pkt2.packet,
            {"username": username, "dept": dept},
            lambda x: f"Email: {x.get('email','?')}",
        )
        if not r2: return

        # Hop 3: Assign licences
        pkt3 = enc.assign_licences(username, licences)
        r3 = bus.dispatch(
            "ORCHESTRATOR", self.it_agent, pkt3.packet,
            {"username": username, "licences": licences},
            lambda x: f"Licences: {x.get('licences_assigned',[])}",
        )

        # Hop 4: Configure access
        pkt4 = enc.configure_access(username)
        r4 = bus.dispatch(
            "ORCHESTRATOR", self.it_agent, pkt4.packet,
            {"username": username},
            lambda x: f"Systems: {x.get('systems_granted',[])}",
        )

        # Hop 5: Send welcome
        pkt5 = enc.send_welcome(username)
        bus.dispatch(
            "ORCHESTRATOR", self.it_agent, pkt5.packet,
            {"username": username},
            lambda x: f"Welcome email sent",
        )

        # Hop 6: Audit
        pkt6 = enc.log_provisioning(username)
        bus.dispatch(
            "ORCHESTRATOR", self.audit_agent, pkt6.packet,
            {"username": username, "workflow": "it_provisioning"},
            lambda x: "Provisioning logged",
        )

        bus.result.outputs = {
            "account": r2, "licences": r3, "access": r4
        }

    # ── Sales qualification workflow ──────────────────────────────────────────

    def _run_sales_qualification(
        self,
        bus:     AACPPacketBus,
        lead_id: str = "L-001",
        lead:    dict = None,
    ):
        """5-hop sales qualification workflow using AACP packets."""
        try:
            from aacp.encoders.workflows.sales import SalesEncoder
            enc = SalesEncoder()
        except ImportError:
            raise ImportError("pip install aacp to use sales workflow")

        lead = lead or {
            "id": lead_id, "company": "Demo Corp",
            "budget_gbp": 75000, "timeline_months": 3,
            "need_score": 8, "authority_score": 9, "engaged": True,
        }

        # Hop 1: Fetch lead
        pkt1 = enc.fetch_lead(lead_id)
        r1 = bus.dispatch(
            "ORCHESTRATOR", self.sales_agent, pkt1.packet,
            {"lead": lead},
            lambda x: str(x.get("lead",{}).get("company","?")),
        )
        if not r1: return

        # Hop 2: BANT score
        pkt2 = enc.score_lead(lead_id)
        r2 = bus.dispatch(
            "ORCHESTRATOR", self.sales_agent, pkt2.packet,
            {"lead": r1, "lead_id": lead_id},
            lambda x: f"Score: {x.get('total_score',0)}/100 — {'QUALIFIED' if x.get('qualified') else 'NOT QUALIFIED'}",
        )
        if not r2: return

        # Hop 3: Route lead
        pkt3 = enc.route_lead(lead_id)
        r3 = bus.dispatch(
            "ORCHESTRATOR", self.sales_agent, pkt3.packet,
            {"score_result": r2, "lead_id": lead_id},
            lambda x: f"Stage: {x.get('stage','?')}",
        )

        # Hop 4: Log
        pkt4 = enc.log_qualification(lead_id)
        bus.dispatch(
            "ORCHESTRATOR", self.audit_agent, pkt4.packet,
            {"lead_id": lead_id},
            lambda x: "Logged",
        )

        # Hop 5: Notify rep
        pkt5 = enc.notify_rep(lead_id)
        r5 = bus.dispatch(
            "ORCHESTRATOR", self.sales_agent, pkt5.packet,
            {"lead_id": lead_id, "routing": r3},
            lambda x: f"Rep notified",
        )

        bus.result.outputs = {
            "lead": r1, "score": r2, "routing": r3
        }
