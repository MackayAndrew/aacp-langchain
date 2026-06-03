from ..agent import AACPAgent

class CSAgent(AACPAgent):
    name   = "CS-AGENT"
    domain = "CS"
    system_prompt = """You are CS-Agent. You understand AACP v1.1 coordination packets.
Respond with valid JSON only. No markdown fences.

FETCH|CS|res:customer_profile — {"customer_id":"string","ltv_gbp":N,"loyalty_years":N,"risk_level":"HIGH|MEDIUM|LOW"}
PROC|CS|res:ticket_triage — {"category":"delivery|billing|product|refund","escalate":true|false}
RESOLVE|CS — {"resolution_strategy":"string","tone":"empathetic","goodwill_offer":true,"goodwill_amount_gbp":N}
SEND|CS — {"sent":true,"customer_id":"string"}
LOG|CS — {"logged":true}
"""
