import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Ensure .env is loaded.
load_dotenv()

# --- Configuration for Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678") # WARNING: Change this in production!

# Initialize Neo4j Driver (this should be done once globally or on app startup)
# The driver manages the connection pool.
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity() # Verify connection on startup
    print(f"DEBUG (neo4j_handler.py): Successfully connected to Neo4j at {NEO4J_URI}")
except Exception as e:
    print(f"ERROR (neo4j_handler.py): Failed to connect to Neo4j: {e}")
    neo4j_driver = None # Set to None if connection fails to avoid further errors

def insert_triplet_to_neo4j(driver, subject, predicate, object):
    """
    Inserts a subject-predicate-object triplet as nodes and a relationship in Neo4j.
    Uses MERGE to create nodes/relationships if they don't exist, preventing duplicates.
    """
    if not driver:
        print("ERROR (neo4j_handler.py): Neo4j driver not initialized. Cannot insert triplet.")
        return

    with driver.session() as session:
        session.execute_write(
            lambda tx: tx.run(
                """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[:RELATION {type: $predicate}]->(o)
                """,
                subject=subject,
                predicate=predicate,
                object=object
            )
        )

def query_neo4j_for_facts(driver, entities):
    """
    Queries Neo4j for facts (triplets) related to the given list of entities.
    """
    if not driver:
        print("ERROR (neo4j_handler.py): Neo4j driver not initialized. Cannot query for facts.")
        return []

    collected_results = set() # Use a set to avoid duplicate facts
    with driver.session() as session:
        for entity_name in entities:
            # Query for nodes whose names contain the entity_name (case-insensitive)
            # and any relationships connected to them.
            print(f"DEBUG (neo4j_handler.py): Querying Neo4j for entity: '{entity_name}'")
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            OPTIONAL MATCH (e)-[r]->(n)
            RETURN e.name AS subject, type(r) AS predicate, n.name AS object
            
            """
            result = session.run(query, name=entity_name)
            for record in result:
                # Format retrieved data as a string triplet for the RAG context
                if record['subject'] and record['predicate'] and record['object']:
                    fact = f"({record['subject']})-[:{record['predicate']}]->({record['object']})"
                    collected_results.add(fact)
    return list(collected_results) # Convert back to list for consistent return type