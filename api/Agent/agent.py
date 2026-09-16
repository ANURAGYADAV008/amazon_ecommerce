from pydantic import BaseModel, Field
from typing import Literal, Annotated, List, Any
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage
import instructor
from langsmith import traceable, get_current_run_tree
from openai import OpenAI
from langchain_core.messages import  convert_to_openai_messages
from  .utils import format_ai_message

class RAGUsedContext(BaseModel):
    id:str=Field(description="The ID Of the item used answer the questions")
    description:str=Field(description="Short description of the item used to answer the Question")


class Toolcall(BaseModel):
    name:str
    arguments:dict

class FinalResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")


class AgentResponse(BaseModel):
    answer:str
    tool_calls:List[Toolcall]=Field(default_factory=list)
    final_answer: bool = Field(description="True if you have all the information needed to provide a complete answer, False otherwise.")
    references: List[RAGUsedContext] = Field(default_factory=list, description="List of items used to answer the question")

class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(description="An answer to the question if it's not relevant, saying that the question is not relevant to the products in stock")


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-4.1-mini",
    },
)
def agent_node(state) -> dict:
    prompt_template = """
You are a shopping assistant that can answer questions about the products in stock.

You will be given a conversation history and a list of tools you can use to answer the latest query.

<Available tools>
{{ available_tools | tojson }}
</Available tools>

When making tool calls, use this exact format:
{
    "name": "tool_name",
    "arguments": {
        "parameter1": "value1",
        "parameter2": "value2"
    }
}

CRITICAL: All parameters must go inside the "arguments" object, not at the top level of the tool call.

Examples:
- Get formatted item context:
{
    "name": "get_formatted_item_context",
    "arguments": {
        "query": "Kool kids toys.",
        "top_k": 5
    }
}

CRITICAL RULES:
- If tool_calls has values, final_answer MUST be false.
- You cannot call tools and exit the graph in the same response.
- If final_answer is true, tool_calls MUST be []
- You must wait for tool results before exiting the graph.
- If you need tool results before answering, set:
  tool_calls=[...], final_answer=false
- After receiving tool results, you can then set:
  tool_calls=[], final_answer=true
- Use names specifically provided in the available tools. Don't add any additional text to the names.

Instructions:
- You need to answer the question based on the outputs from the tools using the available tools only.
- Do not suggest the same tool call more than once.
- If the question can be decomposed into multiple sub-questions, suggest all of them.
- If multiple tool calls can be used at once to answer the question, suggest all of them.
- Do not explain your next steps in the answer, instead use tools to answer the question.
- Never use word context and refer to it as the available products.
- You should only answer questions about the products in stock. If the question is not about the products in stock, you should ask for clarification.

- As an output you need to return the following:
    * answer: The answer to the question based on your current knowledge and the tool results.
    * references: The list of the indexes from the chunks returned from all tool calls that were used to answer the question. If more than one chunk was used to compile the answer from a single tool call, be sure to return all of them.
      Each reference should have an id and a short description of the item based on the retrieved context.
    * final_answer: True if you have all the information needed to provide a complete answer, False otherwise.

- The answer to the question should contain detailed information about the product and should be returned with detailed specification in bullet points.
- The short description should have the name of the item.
- If the user's request requires using a tool, set tool_calls with the appropriate function names and arguments.
"""

    template = Template(prompt_template)

    prompt = template.render(
        available_tools=state.available_tools,
    )

    messages = state.messages

    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[
            {"role": "system", "content": prompt},
            *conversation,
        ],
        temperature=0.5,
    )

    current_run=get_current_run_tree()

    if  current_run:
        current_run.metadata["usage_metadata"]={
        "input_token":response.usage.prompt_tokens,
        "output_token":response.usage.total_tokens,
        "total_token":response.usage.total_tokens
    }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references,
    }


@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider":"openai"}
)
def intent_router_node(state):
    prompt_template=""" You are part of a shopping assistant that can answer questions about product in stock

    Instruction:
    -You are given a question and you need to classify it into relevant or not relavant.
    -If the question is not relevant ,return False in Field ""question_relavant" and set "answer" to explanation why it is not relevant.
    -If the question is relevant , return True in field "question_relavant" and set "answer" to .
    - You should only answer question the products in stock. If the question is not about the products in stock ,you should ask for clarification
    <Question>
    {{query}}
    </Question>
    -
    """
    template = Template(prompt_template)
    prompt = template.render()

    messages = state.messages
    conversation = []
    
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_provider(
        "openai/gpt-5.4-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response,raw_response=client.create_with_completion(
        model="gpt-4.1-mini",
        response_model=IntentRouterResponse,
        messages=[
            {
                "role":"system",

                 "content":prompt,
            },
            *conversation
        ],
        temperature=0.5,
    )

    current_run=get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata']={
            "input_tokens":raw_response.usage.prompt_tokens,
            "output_token":raw_response.usage.completion_tokens,
            "total_token":raw_response.usage.total_token
        }
        trace_id=str(getattr(current_run,"trace_id",current_run.id))
    else:
        trace_id=None

    return {
        "question_relevant":response.question_relevant,
        "answer":response.answer
    }
