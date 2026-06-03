"""
department_day.py
Full department day workflow -- chains 5 sub-workflows.

Mirrors lab v3 scope:
  1. JML Onboarding       — 3 new hires × 6 hops = 18 hops
  2. Payroll              — 5 hops
  3. Sales Qualification  — 3 leads × 5 hops = 15 hops
  4. CS Resolution        — 3 tickets × 5 hops = 15 hops
  5. Month-End Close      — 6 hops
  ─────────────────────────────────────
  Total coordination hops: 59

Without AACP: 59 LLM coordination calls
With AACP:     0 LLM coordination calls
"""

import csv
import json
import time
from pathlib import Path


# ── Mock data ──────────────────────────────────────────────────────────────

MOCK_EMPLOYEES = [
    {"id":"E001","name":"Alice Smith","dept":"Engineering",
     "cost_centre":"CC-10","base_salary_gbp":"72000",
     "delta_gbp":"0","pension_rate":"0.05","status":"active"},
    {"id":"E002","name":"Bob Jones","dept":"Sales",
     "cost_centre":"CC-20","base_salary_gbp":"58000",
     "delta_gbp":"2500","pension_rate":"0.05","status":"active"},
    {"id":"E003","name":"Carol White","dept":"Finance",
     "cost_centre":"CC-30","base_salary_gbp":"65000",
     "delta_gbp":"0","pension_rate":"0.08","status":"active"},
    {"id":"E004","name":"David Brown","dept":"Engineering",
     "cost_centre":"CC-10","base_salary_gbp":"85000",
     "delta_gbp":"5000","pension_rate":"0.05","status":"active"},
    {"id":"E005","name":"Eve Davis","dept":"HR",
     "cost_centre":"CC-40","base_salary_gbp":"52000",
     "delta_gbp":"0","pension_rate":"0.05","status":"active"},
    {"id":"E006","name":"Frank Miller","dept":"Engineering",
     "cost_centre":"CC-10","base_salary_gbp":"91000",
     "delta_gbp":"3000","pension_rate":"0.05","status":"active"},
    {"id":"E007","name":"Grace Lee","dept":"Sales",
     "cost_centre":"CC-20","base_salary_gbp":"54000",
     "delta_gbp":"1500","pension_rate":"0.05","status":"active"},
    {"id":"E008","name":"Henry Wilson","dept":"Finance",
     "cost_centre":"CC-30","base_salary_gbp":"61000",
     "delta_gbp":"0","pension_rate":"0.08","status":"active"},
]

MOCK_BUDGETS = [
    {"cc_id":"CC-10","cc_name":"Engineering","approved_annual_gbp":"420000",
     "ytd_spend_gbp":"378000","owner":"Sarah Chen","gl_code":"GL-1010"},
    {"cc_id":"CC-20","cc_name":"Sales","approved_annual_gbp":"140000",
     "ytd_spend_gbp":"98000","owner":"Marcus Webb","gl_code":"GL-2020"},
    {"cc_id":"CC-30","cc_name":"Finance","approved_annual_gbp":"160000",
     "ytd_spend_gbp":"124000","owner":"David Park","gl_code":"GL-3030"},
    {"cc_id":"CC-40","cc_name":"HR","approved_annual_gbp":"95000",
     "ytd_spend_gbp":"71000","owner":"Linda Torres","gl_code":"GL-4040"},
]

MOCK_NEW_HIRES = [
    {"employee_id":"E009","name":"James Porter","username":"j.porter",
     "dept":"Engineering","role":"Software Engineer",
     "licences":"M365,Slack,GitHub","systems":"email,vpn,sharepoint"},
    {"employee_id":"E010","name":"Priya Sharma","username":"p.sharma",
     "dept":"Sales","role":"Account Executive",
     "licences":"M365,Slack,Salesforce","systems":"email,vpn,salesforce"},
    {"employee_id":"E011","name":"Tom Bradley","username":"t.bradley",
     "dept":"Finance","role":"Finance Analyst",
     "licences":"M365,Slack,NetSuite","systems":"email,vpn,netsuite"},
]

MOCK_LEADS = [
    {"id":"L-001","company":"Apex Systems","budget_gbp":"85000",
     "timeline_months":"3","need_score":"8","authority_score":"9","engaged":"true"},
    {"id":"L-002","company":"Delta Consulting","budget_gbp":"28000",
     "timeline_months":"9","need_score":"4","authority_score":"8","engaged":"false"},
    {"id":"L-003","company":"CoreTech Solutions","budget_gbp":"120000",
     "timeline_months":"2","need_score":"9","authority_score":"10","engaged":"true"},
]

MOCK_TICKETS = [
    {"id":"T-001","customer_id":"C-4421","subject":"Order not arrived",
     "sentiment":"negative","priority":"high","ltv_gbp":"8500","loyalty_years":"4"},
    {"id":"T-002","customer_id":"C-1832","subject":"Wrong item delivered",
     "sentiment":"negative","priority":"medium","ltv_gbp":"2100","loyalty_years":"1"},
    {"id":"T-003","customer_id":"C-9910","subject":"Billing error on invoice",
     "sentiment":"negative","priority":"high","ltv_gbp":"15000","loyalty_years":"7"},
]

MOCK_RULES = {
    "version":"payroll_v2","period":"2026-03","paye_rate":0.20,
    "budget_warning_threshold":0.85,"budget_breach_threshold":0.90,
    "currency":"GBP","pay_date":"2026-03-28",
}


def write_mock_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True)
    for name, rows in [
        ("employees_2026-03.csv", MOCK_EMPLOYEES),
        ("budgets_2026-03.csv",   MOCK_BUDGETS),
        ("new_hires_2026-03.csv", MOCK_NEW_HIRES),
        ("leads_2026-03.csv",     MOCK_LEADS),
        ("tickets_2026-03.csv",   MOCK_TICKETS),
    ]:
        with open(data_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
    with open(data_dir / "payroll_rules.json","w") as f:
        json.dump(MOCK_RULES, f, indent=2)


def run_department_day(bus, agents: dict, data_dir: Path, period: str = "2026-03"):
    """
    Full department day. 5 sub-workflows. ~59 coordination hops.
    Uses AACPPacketBus for coordination throughout.
    Called by both AACPOrchestrator and the comparison baseline.

    agents dict: {
        "hr":      HRAgent,
        "finance": FinanceAgent,
        "it":      ITAgent,
        "sales":   SalesAgent,
        "cs":      CSAgent,
        "audit":   AuditAgent,
    }
    """
    from aacp.encoders.workflows.payroll   import PayrollEncoder
    from aacp.encoders.workflows.jml       import JMLEncoder
    from aacp.encoders.workflows.sales     import SalesEncoder
    from aacp.encoders.workflows.customer_service import CSResolutionEncoder
    from aacp.encoders.workflows.month_end import MonthEndEncoder

    payroll_enc = PayrollEncoder()
    jml_enc     = JMLEncoder()
    sales_enc   = SalesEncoder()
    cs_enc      = CSResolutionEncoder()
    me_enc      = MonthEndEncoder()

    # Load data
    emp_data  = list(csv.DictReader(open(data_dir/"employees_2026-03.csv")))
    bud_data  = list(csv.DictReader(open(data_dir/"budgets_2026-03.csv")))
    hire_data = list(csv.DictReader(open(data_dir/"new_hires_2026-03.csv")))
    lead_data = list(csv.DictReader(open(data_dir/"leads_2026-03.csv")))
    tick_data = list(csv.DictReader(open(data_dir/"tickets_2026-03.csv")))
    rules     = json.load(open(data_dir/"payroll_rules.json"))

    results = {}

    # ── WORKFLOW 1: JML Onboarding (6 hops × 3 hires = 18 hops) ──────────
    print(f"\n  --- Workflow 1/5: JML Onboarding ({len(hire_data)} new hires) ---")
    jml_results = []
    for hire in hire_data:
        uid = hire["username"]
        eid = hire["employee_id"]
        licences = hire.get("licences","M365,Slack").split(",")

        r1 = bus.dispatch("ORCHESTRATOR", agents["hr"],
            jml_enc.fetch_new_hire(eid).packet,
            {"employee": hire},
            lambda x: x.get("employee",{}).get("name","?"))

        r2 = bus.dispatch("ORCHESTRATOR", agents["it"],
            jml_enc.create_account(uid, hire["dept"]).packet,
            {"username": uid, "dept": hire["dept"]},
            lambda x: f"email: {x.get('email','?')}") if r1 else None

        r3 = bus.dispatch("ORCHESTRATOR", agents["it"],
            jml_enc.assign_licences(uid, licences).packet,
            {"username": uid, "licences": licences},
            lambda x: f"licences: {x.get('licences_assigned',[])}") if r2 else None

        r4 = bus.dispatch("ORCHESTRATOR", agents["it"],
            jml_enc.configure_access(uid).packet,
            {"username": uid},
            lambda x: f"systems: {x.get('systems_granted',[])}") if r3 else None

        r5 = bus.dispatch("ORCHESTRATOR", agents["it"],
            jml_enc.send_welcome(uid).packet,
            {"username": uid},
            lambda x: "welcome sent") if r4 else None

        bus.dispatch("ORCHESTRATOR", agents["audit"],
            jml_enc.log_provisioning(uid).packet,
            {"username": uid},
            lambda x: "logged")

        jml_results.append({"hire": hire["name"], "provisioned": r5 is not None})

    results["jml"] = jml_results

    # ── WORKFLOW 2: Payroll (5 hops) ───────────────────────────────────────
    print(f"\n  --- Workflow 2/5: Payroll ---")
    r1 = bus.dispatch("ORCHESTRATOR", agents["hr"],
        payroll_enc.fetch_employees(period).packet,
        {"employees": emp_data, "period": period},
        lambda x: f"{x.get('total_employees',0)} employees")

    r2 = bus.dispatch("ORCHESTRATOR", agents["finance"],
        payroll_enc.fetch_budgets(period).packet,
        {"budgets": bud_data, "period": period},
        lambda x: f"{x.get('flagged_count',0)} flagged") if r1 else None

    r3 = bus.dispatch("ORCHESTRATOR", agents["hr"],
        payroll_enc.merge_and_calculate(period).packet,
        {"employees": r1, "budgets": r2, "rules": rules},
        lambda x: f"£{x.get('totals',{}).get('gross',0):,} gross") if r2 else None

    r4 = bus.dispatch("ORCHESTRATOR", agents["hr"],
        payroll_enc.generate_report(period, period).packet,
        {"payroll_summary": r3, "period": period},
        lambda x: str(x.get("executive_summary",""))[:60]) if r3 else None

    bus.dispatch("ORCHESTRATOR", agents["audit"],
        payroll_enc.log_run(period).packet,
        {"period": period}, lambda x: "logged")

    results["payroll"] = {"total_gross": r3.get("totals",{}).get("gross") if r3 else None}

    # ── WORKFLOW 3: Sales Qualification (5 hops × 3 leads = 15 hops) ──────
    print(f"\n  --- Workflow 3/5: Sales Qualification ({len(lead_data)} leads) ---")
    sales_results = []
    for lead in lead_data:
        lid = lead["id"]

        r1 = bus.dispatch("ORCHESTRATOR", agents["sales"],
            sales_enc.fetch_lead(lid).packet,
            {"lead": lead},
            lambda x: str(x.get("lead",{}).get("company","?")))

        r2 = bus.dispatch("ORCHESTRATOR", agents["sales"],
            sales_enc.score_lead(lid).packet,
            {"lead": r1, "lead_id": lid},
            lambda x: f"score {x.get('total_score',0)}/100") if r1 else None

        r3 = bus.dispatch("ORCHESTRATOR", agents["sales"],
            sales_enc.route_lead(lid).packet,
            {"score_result": r2, "lead_id": lid},
            lambda x: f"stage: {x.get('stage','?')}") if r2 else None

        bus.dispatch("ORCHESTRATOR", agents["audit"],
            sales_enc.log_qualification(lid).packet,
            {"lead_id": lid}, lambda x: "logged")

        bus.dispatch("ORCHESTRATOR", agents["sales"],
            sales_enc.notify_rep(lid).packet,
            {"lead_id": lid, "routing": r3},
            lambda x: "rep notified")

        sales_results.append({
            "company":   lead["company"],
            "qualified": r2.get("qualified") if r2 else None,
            "score":     r2.get("total_score") if r2 else None,
        })

    results["sales"] = sales_results

    # ── WORKFLOW 4: CS Resolution (5 hops × 3 tickets = 15 hops) ──────────
    print(f"\n  --- Workflow 4/5: CS Resolution ({len(tick_data)} tickets) ---")
    cs_results = []
    for ticket in tick_data:
        tid  = ticket["id"]
        cid  = ticket["customer_id"]
        ltv  = int(ticket.get("ltv_gbp", 0) or 0)
        sent = ticket.get("sentiment", "negative")

        r1 = bus.dispatch("ORCHESTRATOR", agents["cs"],
            cs_enc.fetch_customer(cid).packet,
            {"ticket": ticket},
            lambda x: f"ltv £{x.get('ltv_gbp',0):,}")

        r2 = bus.dispatch("ORCHESTRATOR", agents["cs"],
            cs_enc.triage_complaint(tid).packet,
            {"ticket": ticket, "customer": r1},
            lambda x: f"category: {x.get('category','?')}") if r1 else None

        r3 = bus.dispatch("ORCHESTRATOR", agents["cs"],
            cs_enc.resolve_complaint(tid, sentiment=sent,
                ltv=ltv, goodwill=(ltv > 5000)).packet,
            {"ticket": ticket, "triage": r2, "customer": r1},
            lambda x: str(x.get("resolution_strategy",""))[:50]) if r2 else None

        bus.dispatch("ORCHESTRATOR", agents["cs"],
            cs_enc.send_resolution(tid, cid).packet,
            {"ticket_id": tid, "resolution": r3},
            lambda x: "response sent") if r3 else None

        bus.dispatch("ORCHESTRATOR", agents["audit"],
            cs_enc.log_resolution(tid).packet,
            {"ticket_id": tid}, lambda x: "logged")

        cs_results.append({
            "ticket_id": tid,
            "resolved":  r3 is not None,
            "goodwill":  r3.get("goodwill_offer") if r3 else None,
        })

    results["cs"] = cs_results

    # ── WORKFLOW 5: Month-End Close (6 hops) ───────────────────────────────
    print(f"\n  --- Workflow 5/5: Month-End Close ---")
    prev = "2026-02"

    r1 = bus.dispatch("ORCHESTRATOR", agents["finance"],
        me_enc.fetch_trial_balance(period).packet,
        {"period": period},
        lambda x: f"gl entries: {x.get('gl_entries','?')} balanced: {x.get('balanced','?')}")

    r2 = bus.dispatch("ORCHESTRATOR", agents["finance"],
        me_enc.reconcile_bank(period).packet,
        {"trial_balance": r1, "period": period},
        lambda x: f"reconciled: {x.get('reconciled','?')}") if r1 else None

    r3 = bus.dispatch("ORCHESTRATOR", agents["finance"],
        me_enc.post_accruals(period).packet,
        {"period": period},
        lambda x: f"posted {x.get('accruals_posted',0)} accruals") if r2 else None

    r4 = bus.dispatch("ORCHESTRATOR", agents["finance"],
        me_enc.variance_analysis(period, prev).packet,
        {"period": period, "prev": prev, "accruals": r3},
        lambda x: f"material variances: {x.get('material_variances',0)}") if r3 else None

    r5 = bus.dispatch("ORCHESTRATOR", agents["finance"],
        me_enc.generate_management_accounts(period).packet,
        {"variance": r4, "period": period},
        lambda x: str(x.get("executive_summary",""))[:60]) if r4 else None

    bus.dispatch("ORCHESTRATOR", agents["audit"],
        me_enc.log_close_certification(period).packet,
        {"period": period}, lambda x: "certified")

    results["month_end"] = {
        "reconciled":        r2.get("reconciled") if r2 else None,
        "material_variances": r4.get("material_variances") if r4 else None,
    }

    return results
