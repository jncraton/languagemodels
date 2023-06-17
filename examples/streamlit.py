"""A simple inference UI using Streamlit"""

import streamlit as st
import languagemodels as lm

st.title("[languagemodels](https://github.com/jncraton/languagemodels) Demo")

st.text_input("Prompt (passed to `lm.do()`)", key="prompt")

# Prompt LLM to get response
response = lm.do(st.session_state.prompt)

st.write(response)
