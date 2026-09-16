from qdrant_client import QdrantClient, models
from qdrant_client.models import  Match,Filter,Field,FieldCondition,MatchValue
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Literal, Annotated, List, Any
from operator import add
from langsmith import traceable, get_current_run_tree
from  .utils import get_tool_descriptions
from .agent import agent_node,intent_router_node
from .tools import get_formatted_context


class Toolcall(BaseModel):
    name:str
    arguments:dict

class RAGUsedContext(BaseModel):
    id:str=Field(description="The ID Of the item used answer the questions")
    description:str=Field(description="Short description of the item used to answer the Question")

class State(BaseModel):
    messages:Annotated[List[Any],add]=[]
    question_relevant:bool=False 
    iteration:int=0
    answer:str=""
    available_tools:List[dict[str,Any]]=[]
    tool_calls:List[Toolcall]=[]
    final_answer:bool=False
    references:Annotated[List[RAGUsedContext],add]=[]
    trac



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
    "messages":[{"role":"user","content":"can I get earPhones for myself, a labtop bag for my wife and and something cool for my kids"}],
    "available_tools": tools_desc,
    "iteration":0
    }
    config={
        "configurable":{
            "thread_id":thread_id
        }
    }

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
     graph=workflow.compile(checkpointer=checkpointer)
     result=graph.invoke(initial_state,config)
    return result


@traceable(
    name="rag_pipeline_wrapper"
)
def rag_agent_wrapper(question, thread_id,topk=5):
    qdrant_client = QdrantClient(url='http://qdrant:6333')

    result = run_agent(question,thread_id)

    used_context = []

    for reference in result.get('references', []):
        payload = qdrant_client.scroll(
            collection_name='amazon-items-collection-01-hybrid-search',
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key='parent_asin', match=MatchValue(value=reference.id))
                ]
            ),
        )[0][0].payload

        image_url = payload.get('image', '')
        price = payload.get('price', None)
        
        if image_url:
            used_context.append({
                'image_url': image_url,
                'price': price,
                'description': reference.description,
            })
        
    return {
        'answer': result['answer'],
        'used_context': used_context,
    }