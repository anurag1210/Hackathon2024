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
        "DEP_ID",
        "REVIEW_ID"
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use employee
reviews to answer questions about their experience in the firm.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer,
say you don't know.

REVIEW_ID,DEP_ID,REVIEW
1,AWM,The management is very supportive and always encourages personal growth.
2,CCB,I feel the workload is quite heavy, but the team collaboration makes it manageable.
3,CCB,The work environment is very positive, and there are plenty of opportunities for advancement.
4,DWM,I think there should be more transparency in the decision-making process.
5,CCB,The benefits and compensation are fair, but the work-life balance could be better.
6,DWM,I appreciate the regular training sessions which help us stay updated with industry trends.
7,AWM,The company culture is inclusive and welcoming, making it a great place to work.
8,BCG,Sometimes the communication between departments can be improved.
9,BCG,I have a lot of autonomy in my role, which I really enjoy.
10,AWM,The resources provided for project completion are very bad and not very helpful.


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

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt
