import json
from typing import Any, Dict, List, Optional

from Research_Agent.src.config.settings import settings
from Research_Agent.src.tools.scraper_tool import scrape_webpage
from Research_Agent.src.tools.search_tool import web_search


class ResearchAgent:
    """Multi-provider ReAct Research Agent configured via central Settings."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 6,
    ):
        self.max_iterations = max_iterations
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.get_default_model(self.provider)
        self.api_key = settings.get_api_key(self.provider)

        # Setup python tool functions
        self.tool_functions = {
            "web_search": self._exec_web_search,
            "scrape_webpage": self._exec_scrape_webpage,
        }

        self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def _exec_web_search(self, args: Dict[str, Any]) -> str:
        print(f"🔧 [TOOL CALL] web_search: {args}")
        results = web_search(
            query=args.get("query"),
            domain=args.get("domain"),
            num_results=args.get("num_results", 5),
        )
        return json.dumps(results, indent=2)

    def _exec_scrape_webpage(self, args: Dict[str, Any]) -> str:
        print(f"🔧 [TOOL CALL] scrape_webpage: {args}")
        res = scrape_webpage(
            url=args.get("url"),
            max_length=args.get("max_length", 4000),
        )
        return json.dumps(
            {
                "url": res.url,
                "title": res.title,
                "status_code": res.status_code,
                "content": res.content,
            },
            indent=2,
        )

    def run_task(
        self, task_description: str, task_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the agent using the configured provider."""
        system_prompt = (
            "You are an expert Research Agent. Gather accurate facts for the sub-task "
            "using web search and scraper tools. Summarize findings clearly with sources."
        )

        user_prompt = f"SubTask: {task_description}"
        if task_context:
            user_prompt += f"\nContext:\n{task_context}"

        print(f"\n[Agent Started] Provider: {self.provider} | Model: {self.model}")
        print(f"Task: {task_description}")

        if self.provider == "gemini":
            return self._run_gemini_loop(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._run_openai_loop(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._run_anthropic_loop(system_prompt, user_prompt)

    # --- GEMINI IMPLEMENTATION ---
    def _run_gemini_loop(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        from google.genai import types

        tools_list = [web_search, scrape_webpage]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools_list,
        )

        try:
            # Use client.chats.create to handle Automatic Function Calling (AFC) cleanly
            chat = self.client.chats.create(model=self.model, config=config)
            response = chat.send_message(user_prompt)

            print("[Agent Completed] Gemini executed research and tool calls.")
            return {
                "status": "completed",
                "result": response.text,
                "provider": "gemini",
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "provider": "gemini"}

    # --- OPENAI IMPLEMENTATION ---
    def _run_openai_loop(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Performs a web search to find relevant URLs and summaries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "domain": {"type": "string"},
                            "num_results": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scrape_webpage",
                    "description": "Extracts text from a target URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "max_length": {"type": "integer", "default": 4000},
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(1, self.max_iterations + 1):
            print(f"--- Iteration {iteration}/{self.max_iterations} ---")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f"[Action] Call tool: {fn_name}({args})")

                    func = self.tool_functions.get(fn_name)
                    output = func(args) if func else "Tool not found"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    })
            else:
                print("[Agent Completed] OpenAI loop finished.")
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "result": msg.content,
                    "provider": "openai",
                }

        return {"status": "max_iterations_reached", "provider": "openai"}

    # --- ANTHROPIC IMPLEMENTATION ---
    def _run_anthropic_loop(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        tools = [
            {
                "name": "web_search",
                "description": "Performs a web search to find URLs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domain": {"type": "string"},
                        "num_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape_webpage",
                "description": "Fetches page text.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_length": {"type": "integer", "default": 4000},
                    },
                    "required": ["url"],
                },
            },
        ]

        messages = [{"role": "user", "content": user_prompt}]

        for iteration in range(1, self.max_iterations + 1):
            print(f"--- Iteration {iteration}/{self.max_iterations} ---")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "tool_use" or tool_blocks:
                assistant_content = []
                for b in response.content:
                    if b.type == "text":
                        assistant_content.append({"type": "text", "text": b.text})
                    elif b.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "input": b.input,
                        })

                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for b in tool_blocks:
                    func = self.tool_functions.get(b.name)
                    output = func(b.input) if func else "Tool not found"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": output,
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = "".join([b.text for b in response.content if getattr(b, "type", None) == "text"])
                print("[Agent Completed] Anthropic loop finished.")
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "result": final_text,
                    "provider": "anthropic",
                }

        return {"status": "max_iterations_reached", "provider": "anthropic"}