"""
AACPAgent
A LangChain-native agent that receives and responds to AACP packets.

Each agent has:
  - A name and domain (HR, FIN, IT, etc.)
  - A LangChain LLM (any provider)
  - A system prompt that teaches it to understand AACP packets
  - A receive() method that accepts packets + data and returns JSON

The agent does NOT generate AACP packets. It receives them and acts.
Packet generation is the orchestrator's job.
"""

import os
import re
import json
import time
from typing import Any

# Model cost registry (per 1M tokens)
MODEL_COSTS = {
    "gpt-4.1-mini":  0.40,
    "gpt-4.1":       2.00,
    "gpt-4o":        5.00,
    "gpt-4o-mini":   0.15,
    "gpt-3.5-turbo": 0.50,
}

AACP_CONTEXT = """You understand AACP v1.1 pipe-delimited coordination packets.
Format: TASK|DOM|return:AGENT|p:PRIORITY|aacp:1.1|key:value...
Interpret packets directly. Respond with valid JSON only. No markdown fences."""


class AACPAgent:
    """
    Base class for all AACP-aware LangChain agents.
    Subclass this and set name, domain, and system_prompt.
    """

    name:          str = "BASE-AGENT"
    domain:        str = "UNKNOWN"
    system_prompt: str = AACP_CONTEXT

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str = None):
        self.model   = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        self._cpm    = MODEL_COSTS.get(model, 1.0)
        self._setup_langchain()

    def _setup_langchain(self):
        """Initialise LangChain LLM. Lazy import so langchain is optional."""
        try:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                max_tokens=2000,
                temperature=0,
            )
        except ImportError:
            self._llm = None

    def _call_llm(self, system: str, user: str) -> tuple[str, int, int]:
        """Call LLM via LangChain. Returns (text, tokens_in, tokens_out)."""
        if self._llm is None:
            raise ImportError(
                "langchain-openai not installed. "
                "Run: pip install langchain-openai"
            )
        from langchain_core.messages import SystemMessage, HumanMessage

        messages  = [SystemMessage(content=system), HumanMessage(content=user)]
        response  = self._llm.invoke(messages)
        tokens_in  = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
        tokens_out = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
        return response.content, tokens_in, tokens_out

    def receive(self, packet: str, data: dict = None) -> dict:
        """
        Receive an AACP packet and return a structured result.
        This is the standard interface called by AACPPacketBus.
        """
        parts = [f"Coordination packet:\n{packet}"]
        if data:
            parts.append(f"\nData:\n{json.dumps(data, indent=2)}")
        parts.append("\nRespond with valid JSON only. No markdown fences.")
        user_msg = "\n".join(parts)

        start = time.time()
        try:
            raw, tokens_in, tokens_out = self._call_llm(
                self.system_prompt, user_msg
            )
            latency_ms = (time.time() - start) * 1000

            # Strip markdown fences if model adds them despite instructions
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            result   = json.loads(raw.strip())
            cost_usd = (tokens_in + tokens_out) / 1_000_000 * self._cpm

            return {
                "result":     result,
                "tokens_in":  tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": round(latency_ms, 0),
                "cost_usd":   round(cost_usd, 6),
                "error":      None,
            }
        except json.JSONDecodeError as e:
            latency_ms = (time.time() - start) * 1000
            return {
                "result": None, "tokens_in": 0, "tokens_out": 0,
                "latency_ms": round(latency_ms, 0), "cost_usd": 0.0,
                "error": f"JSON parse error: {e}",
            }
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return {
                "result": None, "tokens_in": 0, "tokens_out": 0,
                "latency_ms": round(latency_ms, 0), "cost_usd": 0.0,
                "error": str(e),
            }


class AuditAgent:
    """Deterministic audit agent. No LLM. $0.00."""
    name   = "AUDIT-AGENT"
    domain = "LOG"

    def receive(self, packet: str, data: dict = None) -> dict:
        return {
            "result":     {"logged": True, "ts": time.time()},
            "tokens_in":  0, "tokens_out": 0,
            "latency_ms": 1, "cost_usd": 0.0, "error": None,
        }
