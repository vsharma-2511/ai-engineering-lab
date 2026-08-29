from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    stage_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def log_usage(self, stage: str, prompt_tok: int, completion_tok: int):
        total = prompt_tok + completion_tok
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens += total

        if stage not in self.stage_breakdown:
            self.stage_breakdown[stage] = {
                "prompt": 0,
                "completion": 0,
                "total": 0,
            }

        self.stage_breakdown[stage]["prompt"] += prompt_tok
        self.stage_breakdown[stage]["completion"] += completion_tok
        self.stage_breakdown[stage]["total"] += total

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 50,
            "📊 TOTAL TOKEN USAGE REPORT",
            "=" * 50,
            f"Total Prompt Tokens:     {self.prompt_tokens:,}",
            f"Total Completion Tokens: {self.completion_tokens:,}",
            f"Grand Total Tokens:      {self.total_tokens:,}",
            "-" * 50,
            "Breakdown by Stage:",
        ]
        for stage, usage in self.stage_breakdown.items():
            lines.append(
                f"  • {stage:<25}: {usage['total']:,} tokens "
                f"(Prompt: {usage['prompt']:,} | Output: {usage['completion']:,})"
            )
        lines.append("=" * 50)
        return "\n".join(lines)


# Singleton instance
token_tracker = TokenTracker()