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
    node_label="Workers",
    text_node_properties=[
        "DEP_ID",
        "REVIEW_ID"
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use employee
information from the context to answer questions about any thing related to the employee the firm.
And answer any questions related to reservations, check the Reservations table and Property Table

Use the following context to answer questions.
Be as detailed as possible, but only answers whats asked for, but don't make up any information
that's not from the context. If you don't know an answer,
say you don't know.

Worker details table:
EMP_ID,EMP_NAME,LOCN,MGR,GRADE,LOB
A123456,John Doe,Bouremouth,,CEO,AWM
B123456,Man Doe,Bouremouth,1,MD,BCG
C123456,Bret Lee,Bouremouth,4,MD,BCG
D123456,Sachin T,Chicago,3,VP,CCB
E123456,James Bond,Chicago,7,VP,BCG
F123456,Tim Cook,Chicago,2,ED,CCB
G123456,Naruto Uz,Bengaluru,3,Assoc,AWM
H123456,Stacy Hull,Bengaluru,4,Assoc,CCB
I123456,Ode Ode,Hong Kong,4,Assoc,CCB
J123456,Random,Hong Kong,8,Assoc,DWM

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

worker_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
worker_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt
