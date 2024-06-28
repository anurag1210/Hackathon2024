import os

from chains.hospital_cypher_chain import hospital_cypher_chain
from chains.hospital_review_chain import reviews_vector_chain
from chains.worker_review_chain import worker_vector_chain
from chains.reservation_chain import reservation_chain

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about employee experiences, feelings, or any other qualitative
        question that could be answered about an employee and their reveiews about 
        their department.
        Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "How is the mood across the firm?"
        "Are employees happy in the firm?"
        "What are the experiences of employees in the firm? Are there any bad reviews?"
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
    Tool(
        name="Reservation",
        func=reservation_chain.invoke,
        description="""
        Use when you need to find out about reservations made my employees.
        Also if user asks you to reserve a seat near a specific employee, then wait for 2 secs
        then respond back with a exact seat number from the context provided that you booked. 
        To find a bookable seat, check where the requested employee is seated and get a seat in the same pproeprty and floor
    
        Or if the user asks you to book a particular seat and date then say it's booked it.
        
        This tool does not have any information about aggregate or any other department data.
        """,
    ),
    Tool(
        name="Employee",
        func=worker_vector_chain.invoke,
        description="""
        Use when you need to find out about any information about employees.
        But do not give his review details.
        
        This tool does not have any information about aggregate or any other department data.
        """,
    )
]

chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)

hospital_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
