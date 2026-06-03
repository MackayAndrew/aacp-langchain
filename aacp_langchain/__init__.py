"""
aacp-langchain
AACP coordination layer for LangChain multi-agent workflows.

Quick start:
    from aacp_langchain import AACPOrchestrator

    orch = AACPOrchestrator(model="gpt-4.1-mini")
    result = orch.run_workflow("payroll", period="2026-03")
    print(result.summary())

Comparison demo:
    python3 examples/comparison.py --mock
"""

from .orchestrator          import AACPOrchestrator
from .agent                 import AACPAgent
from .packet_bus            import AACPPacketBus
from .baseline_orchestrator import BaselineOrchestrator

__version__ = "0.1.0"
__all__ = [
    "AACPOrchestrator",
    "AACPAgent",
    "AACPPacketBus",
    "BaselineOrchestrator",
]
