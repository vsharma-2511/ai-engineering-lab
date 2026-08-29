from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from openai import OpenAI

from Research_Agent.src.config.settings import settings
from Research_Agent.src.schemas.query_schema import ExecutionPlan
from Research_Agent.src.utils.token_tracker import token_tracker


class MetricHighlight(BaseModel):
    metric_name: str = Field(description="Name of the key metric (e.g., '2024 CPI Inflation Rate')")
    value: str = Field(description="Extracted value or range (e.g., '2.4%')")
    context: str = Field(description="Brief contextual comparison or trend (e.g., 'Down from 3.9% in 2023')")


class Citation(BaseModel):
    source_title: str = Field(description="Title or publisher of the source (e.g., 'Statistics Canada Daily')")
    url: str = Field(description="Direct web URL of the cited source")


class SynthesizedReport(BaseModel):
    title: str = Field(description="Professional report title summarizing the findings")
    executive_summary: str = Field(description="High-level 2-3 sentence overview answering the core user query")
    key_metrics: List[MetricHighlight] = Field(description="Key metrics and indicators extracted from findings")
    detailed_analysis: str = Field(
        description="Comprehensive Markdown synthesis detailing all sub-task findings, trends, and drivers")
    sources: List[Citation] = Field(description="Distinct list of official sources cited in the analysis")


SYSTEM_PROMPT = """
You are the Lead Research Synthesizer for an evidence-grounded research system.
Your job is to compile isolated research task outputs into a cohesive, publication-ready research report.

Rules:
1. Ground every claim strictly in the provided task execution results. Do not speculate or introduce unverified data.
2. Structure the detailed analysis cleanly using Markdown with explicit sections and inline markdown links where appropriate.
3. Extract accurate key metrics into the structured schema.
4. Eliminate duplicate findings across tasks and harmonize contrasting data points cleanly.
"""


class Synthesizer:
    """Compiles multi-task execution results into a unified, structured final report."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.llm_provider).lower()

    def _format_task_results(self, execution_results: Dict[str, Any]) -> str:
        """Converts raw executor output dictionary into a clean markdown prompt string."""
        formatted_blocks = []
        task_data = execution_results.get("results", {})

        for task_id, res in task_data.items():
            desc = res.get("description", "N/A")
            status = res.get("status", "completed")
            findings = res.get("result", "No findings recorded.")

            block = (
                f"### Node [{task_id}]\n"
                f"**Task Target:** {desc}\n"
                f"**Status:** {status}\n"
                f"**Findings & Sources:**\n{findings}\n"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def _synthesize_with_openai(self, prompt: str) -> SynthesizedReport:
        """Synthesizes report using OpenAI structured outputs."""
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=SynthesizedReport,
        )
        usage = response.usage
        token_tracker.log_usage(
            stage="Synthesizer",  # or "Executor: TASK_1", "Synthesizer"
            prompt_tok=usage.prompt_tokens,
            completion_tok=usage.completion_tokens,
        )
        return response.choices[0].message.parsed

    def _synthesize_with_gemini(self, prompt: str) -> SynthesizedReport:
        """Synthesizes report using Gemini via Chat structured generation."""
        client = genai.Client(api_key=settings.gemini_api_key)
        chat = client.chats.create(
            model=settings.gemini_model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=SynthesizedReport,
            ),
        )
        response = chat.send_message(message=prompt)
        return SynthesizedReport.model_validate_json(response.text)

    def synthesize(
            self,
            user_query: str,
            execution_plan: ExecutionPlan,
            execution_results: Dict[str, Any],
    ) -> SynthesizedReport:
        """Entrypoint to process task findings and produce a SynthesizedReport."""
        task_context = self._format_task_results(execution_results)

        prompt = (
            f"**Original User Query:** {user_query}\n\n"
            f"**Target Parameters:**\n"
            f" • Indicators: {execution_plan.extracted_params.indicators}\n"
            f" • Locations: {execution_plan.extracted_params.locations}\n"
            f" • Timeframe: {execution_plan.extracted_params.timeframe}\n"
            f" • Target Domains: {execution_plan.extracted_params.target_domains}\n\n"
            f"**Completed Task Findings:**\n"
            f"{task_context}\n\n"
            f"Synthesize these findings into a unified, high-quality research report."
        )

        print("⏳ [Synthesizer] Compiling final research report...")

        if self.provider == "gemini":
            return self._synthesize_with_gemini(prompt)
        elif self.provider == "openai":
            return self._synthesize_with_openai(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: '{self.provider}'")