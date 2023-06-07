import streamlit as st
import languagemodels as lm

st.title("Chatbot")


def reset():
    st.session_state.dialog = ""
    st.session_state.message = ""


# Initialize empty dialog context on first run
if "dialog" not in st.session_state:
    reset()

if st.session_state.message:
    # Add new message to dialog
    st.session_state.dialog += f"User: {st.session_state.message}\n\nAssistant: "
    st.session_state.message = ""

    # Prompt LLM to get response
    response = lm.chat(f"{st.session_state.dialog}")

    # Display full dialog
    st.session_state.dialog += response + "\n\n"

    st.write(st.session_state.dialog)

st.text_input("Message", key="message")

st.button("Reset", on_click=reset)
