"""A simple inference UI using Streamlit

A live version of this application is hosted here:

https://jncraton-languagemodels-examplesstreamlit-0h6yr7.streamlit.app/
"""

import streamlit as st
import languagemodels as lm

st.title("[languagemodels](https://github.com/jncraton/languagemodels) Demo")

st.text_input("Prompt (passed to `lm.do()`)", key="prompt")

# Prompt LLM to get response
response = lm.do(st.session_state.prompt)

st.write(response)
