import streamlit as st  # type: ignore[import]
import requests  # type: ignore[import]
import config


st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I assist you today?'}]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

#Display the messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    suggestions_tab, = st.tabs(["Suggestions"])

    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):

                # Description
                st.caption(item.get("description", "No Description"))

                # Image (only if valid)
                image_url = item.get("image_url")
                if image_url is not None and str(image_url).strip() != "":
                    try:
                        st.image(image_url, width=250)
                    except Exception:
                        st.caption("⚠️ Unable to load image.")

                # Price
                price = item.get("price")
                if price is not None:
                    st.caption(f"Price: {price} USD")

                st.divider()
        else:
            st.info("No suggestions yet")


if prompt := st.chat_input("Hi, how can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        state, output = api_call('post', f'{config.API_URL}/rag', json={'query': prompt})
        answer = output['answer']
        used_context = output['used_context']
        st.session_state.used_context = used_context
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

