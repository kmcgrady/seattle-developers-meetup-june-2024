import streamlit as st

st.title('Hello, Streamlit!')

name = st.text_input('What is your name?')

st.write(f"Hello {name}!")
