"""
Tests for AACPPacketBus -- no API calls needed.
Tests the coordination layer without touching LLMs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aacp_langchain.packet_bus import AACPPacketBus, WorkflowResult
from aacp_langchain.agent import AuditAgent

passed = failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ FAIL: {label}")
        failed += 1


class MockAgent:
    name = "MOCK-AGENT"
    def receive(self, packet, data=None):
        return {
            "result": {"ok": True, "packet_received": packet},
            "tokens_in": 50, "tokens_out": 20,
            "latency_ms": 100, "cost_usd": 0.00003,
            "error": None,
        }

class FailAgent:
    name = "FAIL-AGENT"
    def receive(self, packet, data=None):
        return {
            "result": None, "tokens_in": 0,
            "tokens_out": 0, "latency_ms": 10,
            "cost_usd": 0.0, "error": "JSON parse error",
        }


print("\n" + "="*50)
print("  AACPPacketBus Tests")
print("="*50)

import tempfile, shutil
tmpdir = tempfile.mkdtemp()

try:
    bus = AACPPacketBus(
        workflow="test", model="gpt-4.1-mini",
        audit_log=f"{tmpdir}/audit.jsonl",
        verbose=False,
    )

    # Valid packet dispatched correctly
    agent = MockAgent()
    result = bus.dispatch(
        "ORCHESTRATOR", agent,
        "FETCH|HR|return:HR-Agent|p:1|aacp:1.1|res:emp_salary|period:2026-03",
    )
    check("valid packet dispatches", result is not None)
    check("result contains expected key", result.get("ok") is True)
    check("hop recorded", len(bus.result.hops) == 1)
    check("hop is valid", bus.result.hops[0].valid)
    check("cost accumulated", bus.result.total_cost > 0)

    # Invalid packet flagged
    bus2 = AACPPacketBus(
        workflow="test2", model="gpt-4.1-mini",
        audit_log=f"{tmpdir}/audit2.jsonl",
        verbose=False,
    )
    result2 = bus2.dispatch("ORCHESTRATOR", agent, "INVALID_PACKET")
    check("invalid packet hop recorded", len(bus2.result.hops) == 1)
    check("invalid packet flagged", not bus2.result.hops[0].valid)

    # Audit agent works at $0.00
    audit = AuditAgent()
    bus3 = AACPPacketBus(
        workflow="test3", model="gpt-4.1-mini",
        audit_log=f"{tmpdir}/audit3.jsonl",
        verbose=False,
    )
    r3 = bus3.dispatch(
        "ORCHESTRATOR", audit,
        "LOG|HR|return:AUD-Agent|p:2|aacp:1.1|status:complete",
    )
    check("audit agent returns result", r3 is not None)
    check("audit agent costs $0.00", bus3.result.total_cost == 0.0)

    # WorkflowResult summary
    summary = bus.result.summary()
    check("summary is non-empty", len(summary) > 0)
    check("summary contains model", "gpt-4.1-mini" in summary)

    # Audit file written
    import json as _json
    from pathlib import Path
    audit_path = Path(f"{tmpdir}/audit.jsonl")
    check("audit file written", audit_path.exists())
    lines = audit_path.read_text().strip().split("\n")
    check("audit has one record", len(lines) == 1)
    record = _json.loads(lines[0])
    check("audit record has packet field", "packet" in record)

    print(f"\n{'='*50}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")

finally:
    shutil.rmtree(tmpdir)

if failed > 0:
    sys.exit(1)
else:
    print("\n  PacketBus tests passed. No API calls needed.\n")
