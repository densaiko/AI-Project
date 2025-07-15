import streamlit as st

st.title("AI Project in Streamlit")
st.write("Welcome to the AI Project Streamlit application!")
st.markdown("""
- Regression Machine Learning 
- Classification Machine Learning
- RAG AI Application
""")

with st.form("Prediction"):
    brand = st.selectbox("Select Brand", ["Brand A", "Brand B", "Brand C"])
    price = st.number_input("Enter Price", min_value=0, step=1)
    education = st.radio("Select Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    no_training = st.slider("Number of Trainings", min_value=0, max_value=100, step=1)

    submitted = st.form_submit_button("Submit")


# sidebar for navigation
with st.sidebar:
    st.image("AHM.png", width=210)
    st.write("## First RAG AI Application")
    st.markdown(
        """
        - Understanding Splitting and Chunking
        - Understanding Vectorization
        - Understanding Retrieval
        - Understanding LLMs 
        """
    )
    st.markdown("---")

