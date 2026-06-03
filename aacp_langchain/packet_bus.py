"""
AACPPacketBus
The coordination layer between agents in a LangChain workflow.

Replaces LLM-to-LLM natural language with typed AACP packets.
Every packet is validated before dispatch.
Every packet and response is logged for audit.

This is the core of the integration -- the thing that makes
LangChain coordination deterministic instead of probabilistic.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Standalone validator -- no anthropic dependency needed
VALID_TASKS = {
    "FETCH", "PROC", "FLAG", "RESOLVE", "LOG", "SEND",
    "BUILD", "MERGE", "CALC", "REPORT", "ACK", "SYNC",
}
VALID_DOMS = {"HR", "FIN", "SALES", "LEGAL", "IT", "CS", "MKT"}


def _validate_packet(packet: str) -> tuple[bool, list[str]]:
    fields = [f.strip() for f in packet.strip().split("|")]
    errors = []
    if len(fields) < 3:
        return False, ["Too few fields"]
    if fields[0] not in VALID_TASKS:
        errors.append(f"Unknown TASK: {fields[0]}")
    if fields[1] not in VALID_DOMS:
        errors.append(f"Unknown DOM: {fields[1]}")
    keys = {f.split(":", 1)[0].lower() for f in fields[2:] if ":" in f}
    if "return" not in keys:
        errors.append("Missing return:")
    if "aacp" not in keys:
        errors.append("Missing aacp:")
    return len(errors) == 0, errors


class HopRecord:
    """Single coordination hop: one agent sends one packet to another."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        packet: str,
        valid: bool,
        errors: list[str],
        result: Any,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float,
    ):
        self.from_agent  = from_agent
        self.to_agent    = to_agent
        self.packet      = packet
        self.valid       = valid
        self.errors      = errors
        self.result      = result
        self.tokens_in   = tokens_in
        self.tokens_out  = tokens_out
        self.cost_usd    = cost_usd
        self.latency_ms  = latency_ms
        self.ts          = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "ts":          self.ts,
            "from":        self.from_agent,
            "to":          self.to_agent,
            "packet":      self.packet,
            "valid":       self.valid,
            "errors":      self.errors,
            "tokens_in":   self.tokens_in,
            "tokens_out":  self.tokens_out,
            "cost_usd":    self.cost_usd,
            "latency_ms":  self.latency_ms,
        }


class WorkflowResult:
    """Accumulated result of a full multi-agent workflow run."""

    def __init__(self, workflow: str, model: str):
        self.workflow    = workflow
        self.model       = model
        self.hops:       list[HopRecord] = []
        self.outputs:    dict = {}
        self.success     = True
        self.started     = datetime.now(timezone.utc).isoformat()

    def add_hop(self, hop: HopRecord):
        self.hops.append(hop)
        if not hop.valid or hop.result is None:
            self.success = False

    @property
    def total_cost(self) -> float:
        return round(sum(h.cost_usd for h in self.hops), 6)

    @property
    def total_tokens(self) -> int:
        return sum(h.tokens_in for h in self.hops)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(h.latency_ms for h in self.hops), 0)

    def summary(self) -> str:
        lines = [
            f"AACP-LangChain Workflow: {self.workflow}",
            f"  Model:    {self.model}",
            f"  Hops:     {len(self.hops)}",
            f"  Tokens:   {self.total_tokens:,}",
            f"  Cost:     ${self.total_cost:.4f}",
            f"  Time:     {self.total_latency_ms/1000:.1f}s",
            f"  Success:  {self.success}",
        ]
        for h in self.hops:
            tag = "✓" if h.valid and h.result else "✗"
            lines.append(
                f"  {tag} [{h.from_agent}] → [{h.to_agent}]"
                f"  {h.tokens_in}in/{h.tokens_out}out"
                f"  ${h.cost_usd:.4f}"
            )
            lines.append(f"    {h.packet[:72]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "workflow":   self.workflow,
            "model":      self.model,
            "started":    self.started,
            "success":    self.success,
            "hops":       [h.to_dict() for h in self.hops],
            "outputs":    self.outputs,
            "totals": {
                "hops":       len(self.hops),
                "tokens_in":  self.total_tokens,
                "cost_usd":   self.total_cost,
                "latency_ms": self.total_latency_ms,
            },
        }


class AACPPacketBus:
    """
    Routes AACP packets between LangChain agents.

    Validates every packet before dispatch.
    Logs every hop to audit trail.
    Accumulates results into a WorkflowResult.
    """

    def __init__(
        self,
        workflow: str,
        model: str,
        audit_log: str = "output/audit.jsonl",
        verbose: bool = True,
    ):
        self.workflow  = workflow
        self.model     = model
        self.verbose   = verbose
        self.audit_log = Path(audit_log)
        self.result    = WorkflowResult(workflow, model)
        self.audit_log.parent.mkdir(exist_ok=True)

    def _log(self, hop: HopRecord):
        with open(self.audit_log, "a") as f:
            f.write(json.dumps(hop.to_dict()) + "\n")

    def _print_hop(self, hop: HopRecord):
        if not self.verbose:
            return
        schema = "VALID" if hop.valid else f"INVALID: {hop.errors}"
        print(f"\n  ┌─ [{hop.from_agent}] → [{hop.to_agent}]")
        print(f"  │  {hop.packet[:80]}")
        print(f"  │  Schema: {schema}")
        if hop.result is None or not hop.valid:
            print(f"  └─ ERROR")
        else:
            print(
                f"  └─ ✓ {hop.tokens_in}in/{hop.tokens_out}out"
                f"  {hop.latency_ms:.0f}ms  ${hop.cost_usd:.4f}"
            )

    def dispatch(
        self,
        from_agent: str,
        to_agent,           # AACPAgent instance
        packet: str,
        data: dict = None,
        preview_fn=None,
    ) -> dict | None:
        """
        Validate and dispatch one AACP packet to a receiving agent.
        Returns the agent's result dict or None on failure.
        """
        valid, errors = _validate_packet(packet)
        response = to_agent.receive(packet, data or {})

        hop = HopRecord(
            from_agent  = from_agent,
            to_agent    = to_agent.name,
            packet      = packet,
            valid       = valid,
            errors      = errors,
            result      = response.get("result"),
            tokens_in   = response.get("tokens_in", 0),
            tokens_out  = response.get("tokens_out", 0),
            cost_usd    = response.get("cost_usd", 0.0),
            latency_ms  = response.get("latency_ms", 0.0),
        )

        self._print_hop(hop)
        if self.verbose and hop.result and preview_fn:
            preview = preview_fn(hop.result)
            if preview:
                print(f"     ↳ {str(preview)[:80]}")

        self._log(hop)
        self.result.add_hop(hop)
        return hop.result
