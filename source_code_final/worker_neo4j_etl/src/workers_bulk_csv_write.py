import logging
import os

from neo4j import GraphDatabase
# from retry import retry

WORKER_CSV_PATH = "https://github.com/anurag1210/Hackathon2024/blob/main/source_code_final/worker_neo4j_etl/src/workers.csv" #os.getenv("HOSPITALS_CSV_PATH")
PROPERTY_CSV_PATH = "https://github.com/anurag1210/Hackathon2024/blob/main/source_code_final/worker_neo4j_etl/src/property.csv" #os.getenv("PAYERS_CSV_PATH")
RESERVATION_CSV_PATH = "https://github.com/anurag1210/Hackathon2024/blob/main/source_code_final/worker_neo4j_etl/src/reservation.csv" #os.getenv("PHYSICIANS_CSV_PATH")
WORKER_REVIEW_CSV_PATH = "https://github.com/anurag1210/Hackathon2024/blob/main/source_code_final/worker_neo4j_etl/src/worker_review.csv" #os.getenv("PATIENTS_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

NODES = ["Worker", "Property", "Reservation", "Worker_Review"]


def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


# @retry(tries=100, delay=10)
def load_all_graph_from_csv() -> None:
    """Load structured worker CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    # LOGGER.info("Setting uniqueness constraints on nodes")
    # with driver.session(database="neo4j") as session:
    #     for node in NODES:
    #         session.execute_write(_set_uniqueness_constraints, node)

    LOGGER.info("Loading worker nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{WORKER_CSV_PATH}' AS workers
        MERGE (h:Workers {{EMP_ID: workers.EMP_ID,
                            EMP_NAME: workers.EMP_NAME,
                            LOCN: workers.LOCN,
                            MGR: workers.MGR,
                            GRADE: toInteger(workers.GRADE),
                            LOB: workers.LOB
                            }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading PROPERTY nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PROPERTY_CSV_PATH}' AS property
        MERGE (p:Property {{PROPERTY_ID: property.PROPERTY_ID,
                            FLOOR: property.FLOOR,
                            SEAT: property.SEAT}});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading RESERVTION nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{RESERVATION_CSV_PATH}' AS reservation
        MERGE (p:Reservation {{RES_ID: toInteger(reservation.RES_ID),
                            PROPERTY_ID: reservation.PROPERTY_ID,
                            FLOOR: reservation.FLOOR,
                            SEAT: reservation.SEAT,
                            RESERVED_BY: reservation.RESERVED_BY
                            RESERVED_ON: reservation.RESERVED_ON
                            }});
        """
        _ = session.run(query, {})

    # LOGGER.info("Loading visit nodes")
    # with driver.session(database="neo4j") as session:
    #     query = f"""
    #     LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
    #     MERGE (v:Visit {{id: toInteger(visits.visit_id),
    #                         room_number: toInteger(visits.room_number),
    #                         admission_type: visits.admission_type,
    #                         admission_date: visits.date_of_admission,
    #                         test_results: visits.test_results,
    #                         status: visits.visit_status
    #     }})
    #         ON CREATE SET v.chief_complaint = visits.chief_complaint
    #         ON MATCH SET v.chief_complaint = visits.chief_complaint
    #         ON CREATE SET v.treatment_description =
    #         visits.treatment_description
    #         ON MATCH SET v.treatment_description = visits.treatment_description
    #         ON CREATE SET v.diagnosis = visits.primary_diagnosis
    #         ON MATCH SET v.diagnosis = visits.primary_diagnosis
    #         ON CREATE SET v.discharge_date = visits.discharge_date
    #         ON MATCH SET v.discharge_date = visits.discharge_date
    #      """
    #     _ = session.run(query, {})
    
    LOGGER.info("Loading worker review nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{WORKER_REVIEW_CSV_PATH}' AS reviews
        MERGE (r:Review {{REVIEW_ID: toInteger(reviews.REVIEW_ID),
                         DEP_ID: reviews.DEP_ID,
                         REVIEW: reviews.REVIEW
                        }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'RESERVED' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{RESERVATION_CSV_PATH}' AS row
        MATCH (target: `Worker` {{ `EMP_ID`: row.`EMP_ID`}})
        MERGE (source)-[r: `RESERVED`]->(target)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'RECEIVES' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{WORKER_REVIEW_CSV_PATH}' AS reviews
            MATCH (v:Worker {{LOB: reviews.DEP_ID}})
            MATCH (r:Review {{id: toInteger(reviews.REVIEW_ID)}})
            MERGE (v)-[receives:RECEIVES]->(r)
        """
        _ = session.run(query, {})

    # LOGGER.info("Loading 'TREATS' relationships")
    # with driver.session(database="neo4j") as session:
    #     query = f"""
    #     LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
    #         MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
    #         MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
    #         MERGE (p)-[treats:TREATS]->(v)
    #     """
    #     _ = session.run(query, {})

    # LOGGER.info("Loading 'COVERED_BY' relationships")
    # with driver.session(database="neo4j") as session:
    #     query = f"""
    #     LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
    #         MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
    #         MATCH (p:Payer {{id: toInteger(visits.payer_id)}})
    #         MERGE (v)-[covered_by:COVERED_BY]->(p)
    #         ON CREATE SET
    #             covered_by.service_date = visits.discharge_date,
    #             covered_by.billing_amount = toFloat(visits.billing_amount)
    #     """
    #     _ = session.run(query, {})

    # LOGGER.info("Loading 'HAS' relationships")
    # with driver.session(database="neo4j") as session:
    #     query = f"""
    #     LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
    #         MATCH (p:Patient {{id: toInteger(visits.patient_id)}})
    #         MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
    #         MERGE (p)-[has:HAS]->(v)
    #     """
    #     _ = session.run(query, {})

    # LOGGER.info("Loading 'EMPLOYS' relationships")
    # with driver.session(database="neo4j") as session:
    #     query = f"""
    #     LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
    #         MATCH (h:Hospital {{id: toInteger(visits.hospital_id)}})
    #         MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
    #         MERGE (h)-[employs:EMPLOYS]->(p)
    #     """
    #     _ = session.run(query, {})


if __name__ == "__main__":
    load_all_graph_from_csv()
