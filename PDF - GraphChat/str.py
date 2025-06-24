#This is a streamlit application that processes PDF documents to extract triplets and build a knowledge graph using Neo4j, with a chat interface for querying the graph
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate # Keep PromptTemplate for structuring prompts

# NEW: Import OpenAI
from openai import OpenAI

from neo4j import GraphDatabase
import json
from dotenv import load_dotenv
import os
import re # Still useful for general string operations, though less for JSON parsing

load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Ensure this is your correct env var
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = ""
NEO4J_PASSWORD = "" # WARNING: Please use a strong password in production!

# NEW: Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Define your model names clearly
OPENROUTER_MODEL_GRAPH_EXTRACTION = "claude-3.7-sonnet"
OPENROUTER_MODEL_RAG = "meta-llama/llama-3.3-70b-instruct" # Can be different if desired


neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- Helper Functions ---
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return splitter.split_text(text)

def extract_triplets(text_chunk):
    # Re-use PromptTemplate for structuring the prompt
    prompt_template_str = """
    You are an expert in extracting triplets from text.
    Your task is to extract all subject-predicate-object triplets from the provided text chunk.
    Each triplet should be represented as a JSON object with keys "subject", "predicate", and "object".

    Text:
    {chunk}

    Output as a JSON array like:
    [
      {{"subject": "...", "predicate": "...", "object": "..."}},
      ...
    ]

    ---
    IMPORTANT: Your response MUST be a JSON array. Do NOT include explanations, commentary, or code block fences (```).
    ---
    Also try to form relations among the triplets, e.g., if a subject appears in multiple triplets, try to use the same subject name.
    this makes sure that the triplets are consistent and can be used to build a knowledge graph and not too much noise is generated.
    this is important for the next step to work correctly.
    """
    prompt = prompt_template_str.format(chunk=text_chunk)

    content = "" # Initialize for error reporting
    try:
        chat_completion = client.chat.completions.create( # Use 'client' object
            model=OPENROUTER_MODEL_GRAPH_EXTRACTION,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"} # CRITICAL: Request JSON output
        )
        content = chat_completion.choices[0].message.content

        parsed_response = json.loads(content) # Direct JSON load

        # Handle cases where LLM might wrap the list in a "triplets" key
        if isinstance(parsed_response, dict) and "triplets" in parsed_response and isinstance(parsed_response["triplets"], list):
            return parsed_response["triplets"]
        elif isinstance(parsed_response, list):
            return parsed_response
        else:
            st.warning(f"LLM returned unexpected root format for triplets after JSON parse: {content}")
            return []
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error in extract_triplets: {e}. Raw content: {content[:200]}...")
        return []
    except Exception as e:
        st.error(f"Error in extract_triplets: {e}")
        return []

def insert_triplet_to_neo4j(driver, subject, predicate, object):
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

def extract_entities_from_question(question):
    prompt_template_str = """
    You are an expert at identifying key entities and relationships from a user's question to facilitate a knowledge graph lookup.
    Given the user's question, identify the main entities (persons, organizations, concepts, dates, products, etc.) and any explicit or implied relationships they are asking about.

    ONLY return a JSON object with two keys:
    'entities': a list of strings, each being a distinct entity identified.
    'relationships': a list of strings, each being a distinct relationship type or action implied (e.g., 'acquired', 'works for', 'created', 'feature of'). If no specific relationship is clear, return an empty list.

    Example Question: "Who works at Google and what is their role?"
    Example Output: {{"entities": ["Google"], "relationships": ["works at", "role"]}}

    User Question: "{question}"

    ---
    IMPORTANT: Your response MUST be a JSON object, and nothing else. Do NOT include any explanations, code block fences (```), or commentary.
    If no entities or relationships are found, return {{"entities": [], "relationships": []}}.
    ---
    """
    prompt = prompt_template_str.format(question=question)

    response_content = "" # Initialize for error reporting
    try:
        chat_completion = client.chat.completions.create( # Use 'client' object
            model=OPENROUTER_MODEL_GRAPH_EXTRACTION,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"} # CRITICAL: Request JSON output
        )
        response_content = chat_completion.choices[0].message.content

        parsed_response = json.loads(response_content) # Direct JSON load

        # Basic validation of expected keys
        if isinstance(parsed_response, dict) and "entities" in parsed_response and "relationships" in parsed_response:
            return parsed_response
        else:
            st.warning(f"LLM returned unexpected format for entities/relationships after JSON parse: {response_content}")
            return {"entities": [], "relationships": []}

    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error from LLM response for entities: {e}. Raw output: {response_content[:500]}...")
        return {"entities": [], "relationships": []}
    except Exception as e:
        st.error(f"Error calling LLM for entity extraction: {e}. Raw LLM output: {response_content[:500]}...")
        return {"entities": [], "relationships": []}

def query_neo4j_for_facts(driver, entities):
    collected_results = set()
    with driver.session() as session:
        for entity_name in entities:
            # Basic query: find entity and its immediate neighbors
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            OPTIONAL MATCH (e)-[r]->(n)
            RETURN e.name AS subject, type(r) AS predicate, n.name AS object
            LIMIT 20
            """
            result = session.run(query, name=entity_name)
            for record in result:
                # Ensure all parts are not None before adding to fact
                if record['subject'] and record['predicate'] and record['object']:
                    fact = f"({record['subject']})-[:{record['predicate']}]->({record['object']})"
                    collected_results.add(fact)
    return list(collected_results)


def process_pdf_to_graph(pdf_docs):
    raw_text = extract_text_from_pdfs(pdf_docs)
    chunks = chunk_text(raw_text)
    all_triplets = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        triplets = extract_triplets(chunk)
        all_triplets.extend(triplets)
        progress_bar.progress((i + 1) / len(chunks))

    st.success(f"Extracted {len(all_triplets)} potential triplets.")


    if st.button("Generate Graph in Neo4j"):
        if not all_triplets:
            st.warning("No triplets extracted to generate graph. Please check PDF content.")
            return ""

        neo4j_progress_bar = st.progress(0)
        for i, t in enumerate(all_triplets):
            try:
                # Ensure subject, predicate, object are strings before inserting
                sub = str(t.get('subject', ''))
                pred = str(t.get('predicate', ''))
                obj = str(t.get('object', ''))
                
                if sub and pred and obj: # Only insert if all parts are non-empty strings
                    insert_triplet_to_neo4j(neo4j_driver, sub, pred, obj)
                else:
                    st.warning(f"Skipping malformed triplet: {t}")
                    
                neo4j_progress_bar.progress((i + 1) / len(all_triplets))
            except Exception as e:
                st.error(f"Error inserting triplet {t} into Neo4j: {e}")

        st.success("Neo4j Graph Generated successfully!")
    
    return raw_text # Keep returning raw_text for now, though it won't be used for Graph RAG

# --- Streamlit UI ---
st.title("PDF to GraphRAG Chatbot")
st.sidebar.header("Upload PDFs")
pdf_docs = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

doc_text = "" # This will hold the raw text, but won't be the primary RAG context anymore
if pdf_docs:
    doc_text = process_pdf_to_graph(pdf_docs)

# --- Chatbot (Graph RAG) ---
st.subheader("Ask Questions (Graph RAG)")

# Display Chat History
for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

user_question = st.chat_input("Ask a question about the uploaded document:")

if user_question:
    st.chat_message("user").write(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."): # Add a spinner for user experience
        # Step 1: Extract entities from user question
        identified_elements = extract_entities_from_question(user_question)
        identified_entities = identified_elements.get("entities", [])
        identified_relationships = identified_elements.get("relationships", [])

        graph_context = ""
        if identified_entities:
            # Step 2: Query Neo4j for relevant facts
            retrieved_facts = query_neo4j_for_facts(neo4j_driver, identified_entities)

            if retrieved_facts:
                graph_context = "Facts from Knowledge Graph:\n" + "\n".join(retrieved_facts)
                if identified_relationships:
                    graph_context += f"\nUser also implied relationships: {', '.join(identified_relationships)}"
            else:
                graph_context = f"No direct facts found in the knowledge graph for entities: {', '.join(identified_entities)}."
                st.info(graph_context) # Show this to the user for debugging/info
        else:
            graph_context = "No specific entities identified in your question to query the knowledge graph."
            st.info(graph_context) # Show this to the user

        # Step 3: Generate RAG response using graph_context
        if graph_context: # Only generate response if some context exists or entities were identified
            rag_prompt_str = f"""
You are a helpful assistant specialized in providing answers based on the provided context from a knowledge graph.
Answer the user's question truthfully and concisely based ONLY on the following facts.
If the answer cannot be found in the provided facts, state that the information is not available in the provided context.
Do not make up information.

Facts from Knowledge Graph:
{graph_context}

Question:
{user_question}

Answer:
"""
            # Use your 'client' object for the RAG call
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL_RAG,
                messages=[{"role": "user", "content": rag_prompt_str}],
                temperature=0.5, # Matching your previous temperature for RAG
            )
            response_text = response.choices[0].message.content
        else:
            response_text = "I couldn't retrieve enough relevant information from the knowledge graph to answer your question. Please try rephrasing or uploading a document with relevant content."

    st.chat_message("assistant").write(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})