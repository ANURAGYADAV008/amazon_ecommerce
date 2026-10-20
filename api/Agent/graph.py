from qdrant_client import QdrantClient
from qdrant_client.models import Filter,Field,FieldCondition,MatchValue
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, List, Any
from operator import add
from langsmith import traceable
from  .utils import get_tool_descriptions
from .agent import coordinator_agent, product_qna_agent, shopping_cart_agent,coordinator_agent, RAGUsedContext,Delegation,Toolcall
from .tools import get_formatted_context,add_to_shopping_cart,remove_from_cart,get_shopping_cart
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
load_dotenv()
import os

DB_URI = os.environ.get("DB_URI", "postgresql://user:password@localhost:5434/ecommerce")
class CoordinatorAgentProperties(BaseModel):
    iteration:int=0
    final_answer:bool=False
    plan:List[Delegation]=[]
    next_agent:str=""

class AgentProperties(BaseModel):
   iteration:int=0
   available_tools:List[dict[str,Any]]=[]
   tool_calls:List[Toolcall]=[]
   final_answer:bool=False


class State(BaseModel):
    messages:Annotated[List[Any],add]=[]
    question_relevant:bool=False 
    user_intent:str=""
    answer:str=""
    product_qa_agent:AgentProperties=Field(default_factory=AgentProperties)
    references:Annotated[List[RAGUsedContext],add]=[]
    shopping_cart_agent:AgentProperties=Field(default_factory=AgentProperties)
    coordinator_agent:CoordinatorAgentProperties=Field(default_factory=CoordinatorAgentProperties)
    user_id:str=""
    cart_id:str=""


def product_qna_agent_tool_router(state: State) -> str:
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(getattr(state.messages[-1], "tool_calls", [])) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_agent_tool_router(state: State) -> str:
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(getattr(state.messages[-1], "tool_calls", [])) > 0:
        return "tools"
    else:
        return "end"

def coordinator_agent_edge(state: State) -> str:
    if state.coordinator_agent.iteration > 4:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qna_agent":
        return "product_qna_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.coordinator_agent.final_answer or not state.coordinator_agent.next_agent:
        return "end"
    else:
        return "end"


workflow = StateGraph(State)

product_qna_tools = [get_formatted_context]
shopping_cart_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]

product_qna_tool_node = ToolNode(product_qna_tools)
shopping_cart_tool_node = ToolNode(shopping_cart_tools)

workflow.add_node("product_qna_tool_node", product_qna_tool_node)
workflow.add_node("shopping_cart_tool_node", shopping_cart_tool_node)
workflow.add_node("product_qna_agent", product_qna_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("coordinator_agent", coordinator_agent)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qna_agent": "product_qna_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END 
    }
)

workflow.add_conditional_edges(
    "product_qna_agent",
    product_qna_agent_tool_router,
    {
        "tools": "product_qna_tool_node",
        "end": "coordinator_agent" 
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_tool_node",
        "end": "coordinator_agent" 
    }
)

workflow.add_edge("product_qna_tool_node", "product_qna_agent")
workflow.add_edge("shopping_cart_tool_node", "shopping_cart_agent")

graph = workflow.compile()


def run_agent(question:str,thread_id:str)->dict:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "product_qa_agent":{
          "iteration":0,
          "final_answer":False,
          "available_tools":get_tool_descriptions(product_qna_tools),
          "tool_calls":[]
       },

        "shopping_cart_agent":{
           "iteration":0,
           "final_answer":False,
           "available_tools":get_tool_descriptions(shopping_cart_tools),
           "tool_calls":[]
        },

        "coordinator_agent": {
           "iteration": 0,
           "final_answer": False,
           "plan": [],
           "next_agent": ""
       },
       "user_id":thread_id,
       "cart_id":thread_id

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
            rating = payload.get('average_rating', payload.get('rating', None))
            
            if image_url:
                used_context.append({
                    'id': reference.id,
                    'rating': float(rating) if rating is not None else None,
                    'image_url': image_url,
                    'price': price,
                    'description': reference.description,
                })
        except Exception:
            continue
        
    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
        {
            'product_id': item.get('product_id'),
            'price': float(item['price']) if item.get('price') is not None else None,
            'quantity': item.get('quantity', 0),
            'currency': item.get('currency', 'USD'),
            'product_image_url': item.get('product_image_url', ''),
            'total_price': float(item['total_price']) if item.get('total_price') is not None else None,
        }
        for item in shopping_cart
    ]
    print("Ok print from wrapper2", used_context)

    return {
        'answer': result.get('answer', ''),
        'used_context': used_context,
        'trace_id': result.get('trace_id'),
        'shopping_cart': shopping_cart_items,
    }
