import os

from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use employee
information from the context to answer questions about any thing related to the employee reservations in the firm.

And answer any questions related to reservations, check the Reservations table and Property Table

Use the following context to answer questions.
Be as detailed as possible, but only answers whats asked for, but don't make up any information
that's not from the context. If you don't know an answer,
say you don't know.


Reservations table:
RES_ID,PROPERTY_ID,FLOOR,SEAT,RESERVED_BY,RESERVED_ON
1,21,1,21-1-SEAT,A123456,2024-06-24
2,21,2,21-2-SEAT,B123456,2024-06-27
3,43,1,43-1-SEAT,C123456,2024-06-24
4,54,1,54-1-SEAT,D123456,2024-06-20
5,65,1,65-1-SEAT,E123456,2024-06-24


Property Table:
PROPERTY_ID,FLOOR,SEAT
21,1,21-1-SEAT
21,2,21-2-SEAT
21,3,21-3-SEAT
43,1,43-1-SEAT
32,1,32-1-SEAT
32,2,32-2-SEAT
43,1,43-1-SEAT
54,1,54-1-SEAT
65,1,65-1-SEAT


{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reservation_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reservation_chain.combine_documents_chain.llm_chain.prompt = review_prompt
