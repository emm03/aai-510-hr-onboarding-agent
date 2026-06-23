
"""
hr_agent.py

HR Onboarding Insights Agent

This file adapts the Assignment 4 ToolCallingAgent structure for the
HR onboarding final project.

Main changes from Assignment 4:
1. The UltraFeedback system prompt is replaced with an HR onboarding prompt.
2. The UltraFeedback Unity Catalog tools are replaced with HR onboarding tools.
3. Vector search and MCP tools are removed because they are not necessary for
   the HR onboarding final project.
4. A small compatibility monkey-patch is included before importing
   databricks_openai, to avoid the VectorSearchIndex import issue in some
   Databricks environments.
"""

import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client


###############################################################################
# 0. Compatibility monkey-patch
###############################################################################

try:
    import databricks.vector_search.client as _vs_client

    if (
        not hasattr(_vs_client, "VectorSearchIndex")
        and hasattr(_vs_client, "VectorSearchClient")
    ):
        _vs_client.VectorSearchIndex = _vs_client.VectorSearchClient
except Exception:
    pass


from databricks_openai import UCFunctionToolkit


###############################################################################
# 1. LLM endpoint, agent skills, and system prompt
###############################################################################

LLM_ENDPOINT_NAME = "openai-chat-gpt-4o-mini"

SYSTEM_PROMPT = f"""
You are an HR Onboarding Insights Assistant.

Your purpose is to help HR leaders, department managers, and employee success teams
understand onboarding effectiveness using company onboarding survey data.

You have access to approved SQL / Unity Catalog onboarding analytics tools.

Available HR onboarding tools:

- lookup_department_onboarding:
  Use this when the user asks about onboarding metrics for one specific department.

- get_lowest_onboarding_department:
  Use this when the user asks which department has the lowest onboarding performance.

- get_highest_probation_loss_location:
  Use this when the user asks which location has the highest probation loss rate.

- lookup_risk_category_summary:
  Use this when the user asks about one specific risk category, such as High Risk,
  Medium Risk, or Low Risk.

- get_company_onboarding_overview_uc:
  Use this when the user asks for overall company-wide onboarding health.

- get_employee_profile_uc:
  Use this only when the user explicitly asks for a restricted HR review of one
  anonymized employee_key.

Tool Selection Rules:
1. Use a tool whenever the question asks for onboarding data, metrics, summaries, comparisons, rankings, recommendations, or employee profiles.
2. Use lookup_department_onboarding for department-specific questions.
3. Use get_lowest_onboarding_department when asked for the weakest department.
4. Use get_highest_probation_loss_location when asked for the location with the highest probation loss rate.
5. Use lookup_risk_category_summary when asked about one specific risk category.
6. Use get_company_onboarding_overview_uc for company-wide onboarding health.
7. Use get_employee_profile_uc only when an anonymized employee_key is explicitly provided.
8. For recommendation-style questions, retrieve relevant tool output before making recommendations.
9. Do not say you will use a tool unless you actually use the tool before giving the final answer.
10. If no available tool can answer the question, explain that the information cannot be retrieved from the current onboarding tools.

Privacy and Safety Rules:
- Never expose raw employee IDs.
- Never list employees in a risk category.
- Never provide raw employee identifiers, names, or any information that could reconstruct employee identity.
- Do not offer to review multiple employee profiles or identify which employees are high risk.
- Use employee_key only when a single anonymized employee-level review is explicitly requested.
- For employee-level questions, provide only a minimal, anonymized, non-decision-support summary.
- Do not recommend hiring, firing, promotion, compensation, disciplinary action, probation decisions, termination, demotion, or other employment decisions.
- Do not make definitive predictions about individual employees.
- Do not make claims related to protected classes or sensitive personal attributes.
- For privacy-sensitive questions, refuse politely and explain that the agent only supports aggregated onboarding analytics or limited anonymized review.
- Use careful language such as "may indicate risk", "suggests a potential issue", or "could benefit from additional support".

Tool Citation and Data Source Rules:
- For every data-backed response, clearly state the tool or data source used.
- Use this format when a tool is used:
  "Based on [tool name], the data shows..."
- If no tool is used because the question is out of scope or unsafe, state that no HR onboarding tool was used.
- Do not give data-backed conclusions without identifying the tool or data source.

Data Grounding Rules:
- Base conclusions only on data returned by tools.
- Do not invent metrics, percentages, scores, policies, or employee details.
- When tool output includes specific metrics, include those values in the response.
- Avoid generic recommendations without data support.
- For general improvement questions, use company-wide overview and/or risk-category summaries before recommending actions.
- If tool output is empty, unavailable, or unclear, say so instead of guessing.

Recommendation Rules:
- Provide HR recommendations only when supported by tool output.
- Frame recommendations as onboarding support actions, not employment decisions.
- Good examples: improve manager check-ins, clarify role expectations, improve training materials, improve tooling access, increase onboarding pulse surveys, improve data quality review.
- Bad examples: fire, discipline, promote, demote, compensate, terminate, penalize, or make probation decisions about employees or managers.

Out-of-Scope Rules:
- If the user asks about weather, sports, coding, general knowledge, or anything unrelated to HR onboarding analytics, politely explain that the question is outside the scope of this agent.
- Do not answer out-of-scope questions from general model knowledge.
- Do not call onboarding tools for clearly irrelevant questions.
- If the user asks about HR policies such as PTO, benefits, payroll, or insurance, explain that the current dataset contains onboarding survey analytics, not HR policy documents.

Response Style:
- Professional
- Data-driven
- Concise but informative
- Executive-friendly
- Privacy-safe

Always prefer factual tool outputs over assumptions.
"""
###############################################################################
# 2. Tool metadata wrapper
###############################################################################

class ToolInfo(BaseModel):
    """
    Represents one callable tool.
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(
    tool_spec: dict,
    exec_fn_param: Optional[Callable] = None,
) -> ToolInfo:
    """
    Convert a Unity Catalog tool specification into a ToolInfo object.
    """

    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)

        if function_result.error is not None:
            return function_result.error

        return function_result.value

    return ToolInfo(
        name=tool_name,
        spec=tool_spec,
        exec_fn=exec_fn_param or exec_fn,
    )


###############################################################################
# 3. Unity Catalog tools
###############################################################################

UC_TOOL_NAMES = [
    "main.default.lookup_department_onboarding",
    "main.default.get_lowest_onboarding_department",
    "main.default.get_highest_probation_loss_location",
    "main.default.lookup_risk_category_summary",
    "main.default.get_company_onboarding_overview_uc",
    "main.default.get_employee_profile_uc",
]

TOOL_INFOS: list[ToolInfo] = []

uc_function_client = get_uc_function_client()
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)

for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


###############################################################################
# 4. Tool schema sanitizer
###############################################################################

def _sanitize_tool_spec(spec: dict) -> dict:
    """
    Remove JSON-schema keywords that some model endpoints reject.
    """

    import copy

    spec = copy.deepcopy(spec)
    params = spec.get("function", {}).get("parameters") or {}

    if not isinstance(params, dict) or "properties" not in params:
        return spec

    for prop in params.get("properties", {}).values():
        if not isinstance(prop, dict):
            continue

        prop_type = prop.get("type")

        if prop_type == "string":
            for key in ("minLength", "maxLength", "pattern"):
                prop.pop(key, None)

        elif prop_type in ("integer", "number"):
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                prop.pop(key, None)

        elif prop_type == "array":
            for key in ("minItems", "maxItems", "uniqueItems"):
                prop.pop(key, None)

    return spec


###############################################################################
# 5. Tool-calling agent
###############################################################################

class ToolCallingAgent(ResponsesAgent):
    """
    HR onboarding tool-calling agent.
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        return [
            _sanitize_tool_spec(tool_info.spec)
            for tool_info in self._tools_dict.values()
        ]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """
        Execute a selected tool with cleaned arguments.
        """

        sane_args = {
            k: v
            for k, v in (args or {}).items()
            if k and isinstance(k, str)
        }

        name = tool_name.strip().strip('"').strip("'")

        if "<" in name:
            name = name.split("<")[0].strip()

        if name in self._tools_dict:
            return self._tools_dict[name].exec_fn(**sane_args)

        candidates = [k for k in self._tools_dict if name.startswith(k)]

        if candidates:
            best_match = max(candidates, key=len)
            return self._tools_dict[best_match].exec_fn(**sane_args)

        raise KeyError(
            f"Unknown tool: {tool_name!r}. Known tools: {list(self._tools_dict.keys())}"
        )

    def call_llm(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[dict[str, Any], None, None]:

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="PydanticSerializationUnexpectedValue",
            )

            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()

                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:

        try:
            args = json.loads(tool_call.get("arguments") or "{}")
        except Exception:
            args = {}

        result = str(
            self.execute_tool(
                tool_name=tool_call["name"],
                args=args,
            )
        )

        tool_call_output = self.create_function_call_output_item(
            tool_call["call_id"],
            result,
        )

        messages.append(tool_call_output)

        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=tool_call_output,
        )

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 20,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:

        for _ in range(max_iter):
            last_msg = messages[-1]

            if last_msg.get("role", None) == "assistant":
                return

            if last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)

            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages),
                    aggregator=messages,
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                "Max iterations reached. Stopping.",
                str(uuid4()),
            ),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None

        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]

        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs,
        )

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:

        session_id = None

        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = to_chat_completions_input(
            [item.model_dump() for item in request.input]
        )

        if SYSTEM_PROMPT:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
            )

        yield from self.call_and_run_tools(messages=messages)


###############################################################################
# 6. Convenience function
###############################################################################

def create_agent(
    llm_endpoint: str = LLM_ENDPOINT_NAME,
    tools: Optional[list[ToolInfo]] = None,
) -> ToolCallingAgent:
    """
    Create the HR onboarding agent.

    Example:
        import hr_agent
        AGENT = hr_agent.create_agent()
    """

    return ToolCallingAgent(
        llm_endpoint=llm_endpoint,
        tools=tools or TOOL_INFOS,
    )


###############################################################################
# 7. MLflow tracing
###############################################################################

mlflow.openai.autolog()
