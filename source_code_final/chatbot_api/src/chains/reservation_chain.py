import os

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

HOSPITAL_QA_MODEL = "gpt-3.5-turbo-0125" #os.getenv("HOSPITAL_QA_MODEL")
HOSPITAL_CYPHER_MODEL = "gpt-3.5-turbo-0125" #os.getenv("HOSPITAL_CYPHER_MODEL")


NEO4J_URI="neo4j+s://4cc7bb89.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="Od4G5DTW0WpGrsfsBl_eNX5-4whcmGwvdlXfjLKnCeY"

graph = Neo4jGraph(
    url=os.getenv(NEO4J_URI),
    username=os.getenv(NEO4J_USERNAME),
    password=os.getenv(NEO4J_PASSWORD),
)

graph.refresh_schema()

res_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

GIVE CYPHER QUERY IN RESPONSE AS WELL

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database.


The question is:
{question}
"""

# Based on your provided data for `Worker`, `Property`, and `Reservation` nodes, here are several Neo4j queries that can help you gain various insights:

# ### Basic Queries

# 1. **List all workers**:
#    ```cypher
#    MATCH (w:Worker)
#    RETURN w.EMP_ID, w.EMP_NAME, w.LOCN, w.MGR, w.GRADE, w.LOB;
#    ```

# 2. **List all properties**:
#    ```cypher
#    MATCH (p:Property)
#    RETURN p.PROPERTY_ID, p.FLOOR, p.SEAT;
#    ```

# 3. **List all reservations**:
#    ```cypher
#    MATCH (r:Reservation)
#    RETURN r.RES_ID, r.PROPERTY_ID, r.FLOOR, r.SEAT, r.RESERVED_BY, r.RESERVED_ON;
#    ```

# ### Relationship-Based Queries

# 4. **Find all reservations made by a specific worker**:
#    ```cypher
#    MATCH (w:Worker {EMP_ID: 'A123456'})-[:RESERVED]->(r:Reservation)
#    RETURN w.EMP_NAME, r.RES_ID, r.RESERVED_ON;
#    ```

# 5. **Find which worker reserved a specific property seat**:
#    ```cypher
#    MATCH (p:Property {PROPERTY_ID: '21', FLOOR: '1', SEAT: '21-1-SEAT'})<-[:ON]-(r:Reservation)<-[:RESERVED]-(w:Worker)
#    RETURN p.PROPERTY_ID, p.FLOOR, p.SEAT, w.EMP_NAME, r.RESERVED_ON;
#    ```

# 6. **Find all properties reserved by workers in a specific location**:
#    ```cypher
#    MATCH (w:Worker {LOCN: 'Chicago'})-[:RESERVED]->(r:Reservation)-[:ON]->(p:Property)
#    RETURN w.EMP_NAME, p.PROPERTY_ID, p.FLOOR, p.SEAT, r.RESERVED_ON;
#    ```

# ### Aggregation Queries

# 7. **Count the number of reservations per worker**:
#    ```cypher
#    MATCH (w:Worker)-[:RESERVED]->(r:Reservation)
#    RETURN w.EMP_NAME, COUNT(r) AS ReservationCount
#    ORDER BY ReservationCount DESC;
#    ```

# 8. **Count the number of reservations per property**:
#    ```cypher
#    MATCH (p:Property)<-[:ON]-(r:Reservation)
#    RETURN p.PROPERTY_ID, p.FLOOR, p.SEAT, COUNT(r) AS ReservationCount
#    ORDER BY ReservationCount DESC;
#    ```

# ### Advanced Queries

# 9. **Find the most reserved properties**:
#    ```cypher
#    MATCH (p:Property)<-[:ON]-(r:Reservation)
#    RETURN p.PROPERTY_ID, p.FLOOR, p.SEAT, COUNT(r) AS ReservationCount
#    ORDER BY ReservationCount DESC
#    LIMIT 5;
#    ```

# 10. **Find workers who have reserved properties across different locations**:
#     ```cypher
#     MATCH (w:Worker)-[:RESERVED]->(r:Reservation)-[:ON]->(p:Property)
#     WITH w, COLLECT(DISTINCT p.LOCN) AS Locations
#     WHERE SIZE(Locations) > 1
#     RETURN w.EMP_NAME, Locations;
#     ```

# 11. **Find all workers who reserved properties on a specific date**:
#     ```cypher
#     MATCH (r:Reservation {RESERVED_ON: '2024-06-24'})<-[:RESERVED]-(w:Worker)
#     RETURN w.EMP_NAME, r.RES_ID, r.PROPERTY_ID, r.FLOOR, r.SEAT;
#     ```

# ### Combining Data Insights



# 13. **Find the managers and the reservations made by their direct reports**:
#     ```cypher
#     MATCH (m:Worker)<-[:MGR]-(w:Worker)-[:RESERVED]->(r:Reservation)
#     RETURN m.EMP_NAME AS Manager, w.EMP_NAME AS Worker, r.RES_ID, r.PROPERTY_ID, r.FLOOR, r.SEAT, r.RESERVED_ON;
#     ```

# ### Example Scenario-Based Queries

# 14. **Find the reservation history for a specific property seat over time**:
#     ```cypher
#     MATCH (p:Property {PROPERTY_ID: '21', FLOOR: '1', SEAT: '21-1-SEAT'})<-[:ON]-(r:Reservation)<-[:RESERVED]-(w:Worker)
#     RETURN p.PROPERTY_ID, p.FLOOR, p.SEAT, w.EMP_NAME, r.RESERVED_ON
#     ORDER BY r.RESERVED_ON;
#     ```

# 15. **Find which workers have reserved multiple different properties**:
#     ```cypher
#     MATCH (w:Worker)-[:RESERVED]->(r:Reservation)-[:ON]->(p:Property)
#     WITH w, COLLECT(DISTINCT p.PROPERTY_ID) AS Properties
#     WHERE SIZE(Properties) > 1
#     RETURN w.EMP_NAME, Properties;
#     ```

# These queries provide a variety of ways to explore the relationships and insights within your Neo4j database. Adjust the queries as needed to fit your specific data and analysis requirements.

# 12. **Find workers and the properties they reserved, grouped by location**:
#     ```cypher
#     MATCH (w:Worker)-[:RESERVED]->(r:Reservation)-[:ON]->(p:Property)
#     RETURN w.LOCN, w.EMP_NAME, COLLECT({PropertyID: p.PROPERTY_ID, Floor: p.FLOOR, Seat: p.SEAT, ReservedOn: r.RESERVED_ON}) AS Reservations
#     ORDER BY w.LOCN, w.EMP_NAME;
#     ```

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=res_generation_template
)

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a users natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When names are provided in the query results, such as hospital names,
beware  of any names that have commas or other punctuation in them.
For instance, 'Jones, Brown and Murray' is a single hospital name,
not multiple hospitals. Make sure you return any list of names in
a way that isn't ambiguous and allows someone to tell what the full
names are.

Never say you don't have the right information if there is data in
the query results. Make sure to show all the relevant query results
if you're asked.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)


reservation_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=HOSPITAL_CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=False,
    top_k=100,
)
