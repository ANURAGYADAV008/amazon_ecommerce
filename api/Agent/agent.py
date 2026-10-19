from pydantic import BaseModel, Field
from typing import Literal, Annotated, List, Any
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
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

class Delegation(BaseModel):
    agent:str
    task:str

class CoordinatorAgent(BaseModel):
    next_agent: str = Field(default="", description="The next agent to invoke: 'product_qna_agent', 'shopping_cart_agent', or '' if no agent is needed.")
    plan: List[Delegation] = Field(default_factory=list, description="Plan of delegations.")
    final_answer: bool = Field(default=False, description="True if no more agent delegation is needed and you are answering the user directly. False when delegating to product_qna_agent or shopping_cart_agent.")
    answer: str = Field(default="", description="The final answer to the user. Leave empty when delegating to another agent.")

class AgentResponse(BaseModel):
    answer:str
    tool_calls:List[Toolcall]=Field(default_factory=list)
    final_answer: bool = Field(description="True if you have all the information needed to provide a complete answer, False otherwise.")
    references: List[RAGUsedContext] = Field(default_factory=list, description="List of items used to answer the question")

class ProductQnAgentResponse(BaseModel):
    answer:str
    tool_calls:List[Toolcall]=Field(default_factory=list)
    final_answer: bool = Field(description="True if you have all the information needed to provide a complete answer, False otherwise.")
    references: List[RAGUsedContext] = Field(default_factory=list, description="List of items used to answer the question")

class ShoppingCartAgentResponse(BaseModel):
    answer:str
    tool_calls:List[Toolcall]=Field(default_factory=list)
    final_answer: bool = Field(description="True if you have all the information needed to provide a complete answer, False otherwise.")
    references: List[RAGUsedContext] = Field(default_factory=list, description="List of items used to answer the question")



@traceable(
    name="product_qua_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-4o-mini",
    },
)

def product_qna_agent(state) -> dict:
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
- Get formatted context:
{
    "name": "get_formatted_context",
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
        available_tools=state.product_qa_agent.available_tools,
    )

    messages = state.messages
   
    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

 

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=ProductQnAgentResponse,
        messages=[
            {"role": "system", "content": prompt},
            *conversation,
        ],
        temperature=0.5,
    )

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "product_qa_agent":{
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools":state.product_qa_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references,
    }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-4o-mini",
    },
)
def shopping_cart_agent(state) -> dict:
    prompt_template = """
You are a part of the shopping assistant that can manage the user's shopping cart.

    ## Instructions

    - Use names specificaly provided in the available tools. Don't add any additional text to the names.
    - You can run multipple tools at once.
    - As the final answer you should return an answer in a form of actions performed.

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

-Add item to shopping cart:
{
   "name":"add_to_shopping_cart",
   "arguments":{
        "items":[
        { "product_id":"123",
         "quantity":1
         
        }
        
        ],
        "user_id":"123",
        "cart_id":"456"
   }
}

-Get shopping Cart:
{
   "name":"get_shopping_cart",
   "arguments":{
     "user_id":"123",
     "cart_id":"456"
   }


}

## Additional information about the user
- User ID: {{ user_id }}
- Cart ID: {{ cart_id }}

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
        available_tools=state.shopping_cart_agent.available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages

    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))


    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=ShoppingCartAgentResponse,
        messages=[
            {"role": "system", "content": prompt},
            *conversation,
        ],
        temperature=0.5,
    )

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "shopping_cart_agent":{
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools":state.shopping_cart_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references,
    }



@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider":"openai"}
)
def coordinator_agent(state):
    
    prompt_template="""  You are a Coordinator Agent as part of a shopping assistant.

    ## Instructions

    - Your role is to understand user queries, create plans for solving user queries, and delegate tasks to specialist agents.
    - When a user asks about products, recommendations, specifications, or inventory availability (including broad requests like "can i get some earphones" or "show me shoes"), ALWAYS delegate immediately to product_qna_agent with next_agent="product_qna_agent", final_answer=False, and answer="". Do not ask for clarification or specify brands/types yourself; let product_qna_agent search and present options.
    - When a user asks to add items to cart, remove items from cart, or view their shopping cart, delegate to shopping_cart_agent with next_agent="shopping_cart_agent", final_answer=False, and answer="".
    - Do NOT reply to the user with promises like "I will find..." when delegating; leave answer empty and simply delegate.
    - Once a specialist agent finishes its task and provides detailed product specifications or cart confirmations, review if any remaining tasks in the plan are needed. If done, set final_answer=True and next_agent="". Do NOT replace, summarize away, or discard the detailed product specifications provided by product_qna_agent.
    - Only answer directly (final_answer=True, next_agent="") if the user's query is a simple greeting, thank you, or completely unrelated to shopping.

    ## Available Agents

    - product_qna_agent: The user is asking a question about a product, specifications, reviews, or available products.
    - shopping_cart_agent: The user is asking to add or remove items from the shopping cart or view cart contents.

    ## Examples

    Question: "Do you have running shoes under $100?"
    next_agent: "product_qna_agent"
    final_answer: false
    answer: ""

    Question: "Can you list the items in my cart?"
    next_agent: "shopping_cart_agent"
    final_answer: false
    answer: ""

    Question: "Hi there!"
    next_agent: ""
    final_answer: true
    answer: "Hello! How can I help you today?"
    """
    template = Template(prompt_template)
    prompt = template.render()

    messages = state.messages
    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))   
    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=CoordinatorAgent,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            *conversation
        ],
        temperature=0.5,
    )

    # If specialist agent (e.g. product_qna_agent) already provided detailed answer, preserve it!
    if state.answer and (response.next_agent in ["", "end"] or not response.answer.strip()):
        chosen_answer = state.answer
    elif response.final_answer and response.answer:
        chosen_answer = state.answer if (state.answer and not response.answer.strip()) else response.answer
    else:
        chosen_answer = state.answer if state.answer else response.answer

    if response.final_answer:
        ai_message = [AIMessage(content=chosen_answer)]
    else:
        ai_message = []

    return {
        "messages": ai_message,
        "user_intent": "",
        "answer": chosen_answer,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": [item.model_dump() for item in response.plan]
        }
    }

# @traceable(
#     name="intent_router_node",
#     run_type="llm",
#     metadata={"ls_provider":"openai"}
# )
# def intent_router_node(state):
#     prompt_template=""" You are part of a shopping assistant that can answer questions about product in stock

#     Instruction:
#     -You are given a question and you need to classify it into relevant or not relavant.
#     -If the question is not relevant ,return False in Field ""question_relavant" and set "answer" to explanation why it is not relevant.
#     -If the question is relevant , return True in field "question_relavant" and set "answer" to .
#     - You should only answer question the products in stock. If the question is not about the products in stock ,you should ask for clarification
#     <Question>
#     {{query}}
#     </Question>
#     -
#     """
#     messages = state.messages
#     query_text = ""
#     if messages:
#         last_msg = messages[-1]
#         if isinstance(last_msg, dict):
#             query_text = last_msg.get("content", "")
#         elif hasattr(last_msg, "content"):
#             query_text = str(getattr(last_msg, "content", ""))

#     template = Template(prompt_template)
#     prompt = template.render(query=query_text)

#     conversation = []
#     for message in messages:
#         msg = convert_to_openai_messages(message)
#         if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"] and msg.get("content"):
#             conversation.append({"role": msg["role"], "content": str(msg["content"])})

#     client = instructor.from_openai(OpenAI())

#     response, raw_response = client.chat.completions.create_with_completion(
#         model="gpt-4.1-mini",
#         response_model=IntentRouterResponse,
#         messages=[
#             {
#                 "role": "system",
#                 "content": prompt,
#             },
#             *conversation
#         ],
#         temperature=0.5,
#     )

#     current_run = get_current_run_tree()
#     if current_run and getattr(raw_response, 'usage', None):
#         usage = raw_response.usage
#         current_run.metadata['usage_metadata'] = {
#             "input_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
#             "output_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)),
#             "total_tokens": getattr(usage, "total_tokens", 0)
#         }
#         trace_id = str(getattr(current_run, "trace_id", current_run.id))
#     else:
#         trace_id = None

#     return {
#         "question_relevant":response.question_relevant,
#         "answer":response.answer,
#         "trace_id":trace_id
#     }

