import sys
from pathlib import Path

# Ensure project root is in sys.path so streamlit_app package is resolvable regardless of execution context
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import uuid
import json
from dotenv import load_dotenv

from streamlit_app.config import API_URL

load_dotenv()

st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

thread_id = get_session_id()

def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        st.session_state['error_popup'] = {
            "visible": True,
            "message": message,
        }
    
    try:
        response = getattr(requests, method)(url, **kwargs)
        
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {'message':'Invalid response format from server'}
        
        if response.ok:
            return True, response_data
        
        return False, response_data
    except requests.exceptions.ConnectionError:
        _show_error_popup('Failed to connect to the server')
        return False, {'message':'Failed to connect to the server'}
    except requests.exceptions.Timeout:
        _show_error_popup('Request timed out')
        return False, {'message':'Request timed out'}
    except Exception as e:
        _show_error_popup(f'An error occurred: {e}')
        return False, {'message':f'An error occurred: {e}'}

def api_call_stream(method, url, **kwargs):

    def _show_error_popup(message):
        st.session_state['error_popup'] = {
            "visible": True,
            "message": message,
        }
    
    try:
        response = getattr(requests, method)(url, **kwargs)
        
        return response.iter_lines()
    except requests.exceptions.ConnectionError:
        _show_error_popup('Failed to connect to the server')
        return False, {'message':'Failed to connect to the server'}
    except requests.exceptions.Timeout:
        _show_error_popup('Request timed out')
        return False, {'message':'Request timed out'}
    except Exception as e:
        _show_error_popup(f'An error occurred: {e}')
        return False, {'message':f'An error occurred: {e}'}
    
def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None 
    
    feedback_data = {
        "run_id": st.session_state.trace_id,
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "feedback_source_type": "api"
    }
    
    status, response = api_call("post", f"{API_URL}/submit_feedback", json=feedback_data)
    return status, response

if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I assist you today?'}]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = thread_id

if "trace_id" not in st.session_state:
    st.session_state.trace_id = ""

if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = None

# Initialize feedback states (simplified)
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

#Display the messages
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

with st.sidebar:
    suggestions_tab, shopping_cart_tab = st.tabs(["🔍 Suggestions", "🛒 Shopping Cart"])
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get('description', 'No Description'))
                if 'image_url' in item:
                    st.image(item['image_url'], width=250)
                st.caption(f"Price: {item['price']} USD")
                st.divider()
        else:
            st.info("No suggestions yet")
    with shopping_cart_tab:
        if st.session_state.shopping_cart:
            for idx, item in enumerate(st.session_state.shopping_cart):
                st.caption(item.get('description', 'No description'))
                if 'product_image_url' in item:
                    st.image(item["product_image_url"], width=250)
                st.caption(f"Price: {item['price']} {item['currency']}")
                st.caption(f"Quantity: {item['quantity']}")
                st.caption(f"Total price: {item['total_price']} {item['currency']}")
                st.divider()
        else:
            st.info("Your cart is empty")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = (
            message["role"] == "assistant" and 
            idx == len(st.session_state.messages) - 1 and 
            idx > 0
        )
        
        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_key = f"feedback_{len(st.session_state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)
            
            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"
                
                # Only submit if this is a new/different feedback
                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(feedback_type=feedback_type)
                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = (feedback_type == "negative")
                        else:
                            st.session_state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()
            
            # Show feedback status message
            if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")
            
            # Show feedback text box if thumbs down was pressed
            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption("Your negative feedback has already been recorded. You can optionally provide additional details below.")
                
                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key=f"feedback_text_{len(st.session_state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100
                )
                
                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button("Send Additional Details", key=f"send_additional_{len(st.session_state.messages)}"):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(feedback_text=feedback_text)
                                if status:
                                    st.success("✅ Thank you! Your additional feedback has been recorded.")
                                    st.session_state.show_feedback_box = False
                                else:
                                    st.error("❌ Failed to submit additional feedback. Please try again.")
                        else:
                            st.warning("Please enter some feedback text before submitting.")
                        st.rerun()
                
                with col_close:
                    if st.button("Close", key=f"close_feedback_{len(st.session_state.messages)}"):
                        st.session_state.show_feedback_box = False
                        st.rerun()

if prompt := st.chat_input("Hi, how can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching inventory and generating answer..."):
            state, output = api_call(
                "post",
                f"{API_URL}/rag",
                json={"query": prompt, "thread_id": st.session_state.thread_id}
            )

        if state and isinstance(output, dict) and "answer" in output:
            answer = output["answer"]
            used_context = output.get("used_context", [])
            trace_id = output.get("trace_id", "")

            st.session_state.used_context = used_context
            st.session_state.trace_id = trace_id
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.session_state.latest_feedback = None
            st.session_state.show_feedback_box = False
            st.session_state.feedback_submission_status = None

            st.markdown(answer)
        else:
            error_msg = (
                output.get("message", output.get("detail", "Unknown error occurred"))
                if isinstance(output, dict)
                else str(output)
            )
            st.error(f"Error from server: {error_msg}")

    st.rerun()