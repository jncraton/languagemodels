import streamlit as st
import languagemodels as lm

st.title("Chatbot")


def reset():
    st.session_state.dialog = "<|system|>Assistant is happy and helpful<|endoftext|>\n\n"
    st.session_state.message = ""


# Initialize empty dialog context on first run
if "dialog" not in st.session_state:
    reset()

if st.session_state.message:
    # Add new message to dialog
    st.session_state.dialog += f"<|prompter|>{st.session_state.message}<|endoftext|>\n\n<|assistant|>"
    st.session_state.message = ""

    # Prompt LLM to get response
    response = lm.chat(
        f"<|system|>Assistant is an AI who is kind honest and helpful<|endoftext|>\n\n"
        f"{st.session_state.dialog}"
    )

    # Display full dialog
    st.session_state.dialog += response + "<|endoftext|>\n\n"

    st.write(st.session_state.dialog)

st.text_input("Message", key="message")

st.button("Reset", on_click=reset)
