from qdrant_client import QdrantClient
from qdrant_client.models import Filter,Field,FieldCondition,MatchValue
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, List, Any
from operator import add
from langsmith import traceable
from  .utils import get_tool_descriptions
from .agent import agent_node, intent_router_node, Toolcall, RAGUsedContext
from .tools import get_formatted_context
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
load_dotenv()
import os

DB_URI = os.environ.get("DB_URI", "postgresql://user:password@localhost:5434/ecommerce")

class State(BaseModel):
    messages:Annotated[List[Any],add]=[]
    question_relevant:bool=False 
    iteration:int=0
    answer:str=""
    available_tools:List[dict[str,Any]]=[]
    tool_calls:List[Toolcall]=[]
    final_answer:bool=False
    references:Annotated[List[RAGUsedContext],add]=[]
    trace_id:str=""



def tool_router(state: State) -> str:
    """Decide Whether to Continue or end"""
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls)>0:
        return "tools"
    else :
        return "end"
  
def intent_router_conditional_edges(state: State) -> str:
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

workflow = StateGraph(State)

tools = [get_formatted_context]

tool_node = ToolNode(tools)

workflow.add_node("tool_node", tool_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END 
    }
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END 
    }
)

workflow.add_edge("tool_node", "agent_node")

graph = workflow.compile()

def run_agent(question:str,thread_id:str)->dict:
    tools_desc=get_tool_descriptions(tools)
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "available_tools": tools_desc,
        "iteration": 0,
    }
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    db_uri = os.environ.get("DB_URI", DB_URI)
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()
        graph = workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config)
    return result


@traceable(
    name="rag_pipeline_wrapper"
)
def rag_agent_wrapper(question, thread_id, topk=5):
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6335"))

    result = run_agent(question,thread_id)

    used_context = []

    for reference in result.get('references', []):
        try:
            scroll_results, _ = qdrant_client.scroll(
                collection_name=os.getenv("QDRANT_COLLECTION_NAME", "Amazon-items-collection-02-hybrid-serach"),
                with_payload=True,
                with_vectors=False,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key='parent_asin', match=MatchValue(value=reference.id))
                    ]
                ),
            )

            if not scroll_results:
                continue

            payload = scroll_results[0].payload or {}
            image_url = payload.get('image', payload.get('image_url', ''))
            price = payload.get('price', None)
            
            if image_url:
                used_context.append({
                    'image_url': image_url,
                    'price': price,
                    'description': reference.description,
                })
        except Exception:
            continue
        
    return {
        'answer': result.get('answer', ''),
        'used_context': used_context,
    }