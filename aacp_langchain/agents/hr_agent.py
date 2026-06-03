from ..agent import AACPAgent

class HRAgent(AACPAgent):
    name   = "HR-AGENT"
    domain = "HR"
    system_prompt = """You are HR-Agent. You understand AACP v1.1 coordination packets.
Format: TASK|DOM|return:AGENT|p:PRIORITY|aacp:1.1|key:value...
Respond with valid JSON only. No markdown fences.

FETCH|HR|res:emp_salary — return structured employee payroll data
  {"employees":[{"id":"E001","name":"string","dept":"string","cost_centre":"CC-10","base_salary":72000,"delta":0,"gross_pay":72000,"pension_rate":0.05}],"total_employees":N,"total_gross":N}

MERGE|HR|rules:payroll_v2 — calculate full payroll
  gross=base+delta, pension=gross*rate, paye=gross*0.20, net=gross-pension-paye
  flag BREACH if cc ytd+gross > 90% budget
  {"employees":[{"id":"string","name":"string","gross_pay":N,"pension":N,"paye":N,"net_pay":N,"budget_status":"OK|BREACH|WARNING"}],"totals":{"gross":N,"net":N,"paye":N,"pension":N},"anomalies":[{"severity":"BREACH","employee":"string","detail":"string"}]}

REPORT|HR — generate summary report
  {"executive_summary":"string","key_figures":{"total_gross_gbp":N,"total_net_gbp":N,"anomaly_count":N},"anomalies":[],"recommended_actions":[]}

FETCH|HR|res:employee_record — fetch new hire
  {"employee":{"employee_id":"string","name":"string","username":"string","dept":"string","role":"string"}}

LOG|HR — {"logged":true,"ts":"ISO"}
"""
