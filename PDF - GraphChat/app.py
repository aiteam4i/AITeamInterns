import os
import json
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify

# Import functions from your services modules
# Make sure these paths are correct relative to app.py
from services.llm_service import extract_triplets, extract_entities_from_question, get_rag_response
from services.neo4j_handler import neo4j_driver, insert_triplet_to_neo4j, query_neo4j_for_facts
from services.pdf_processor import extract_text_from_pdfs, chunk_text

# Load environment variables from .env file
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)
# No app.secret_key needed as Flask session is not used for state management

# --- Global In-memory Storage (Volatile: resets on server restart) ---
# For demonstration/debugging without Flask session or a full database.
# In a production multi-user app, you'd use a database for persistent state.
temp_extracted_triplets = [] # Stores triplets after PDF processing, before Neo4j insertion
temp_uploaded_files_info = [] # Stores names of uploaded files for display

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main page.
    Passes initial data (like uploaded files) to the template from global memory.
    Chat history is handled client-side in this no-session setup.
    """
    return render_template('index.html', chat_history=[], uploaded_files=temp_uploaded_files_info)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handles PDF file uploads.
    Extracts text, chunks it, and extracts triplets using LLM.
    Stores extracted triplets temporarily in global memory.
    """
    global temp_extracted_triplets
    global temp_uploaded_files_info

    if 'pdf_files' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files or pdf_files[0].filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    uploaded_file_names = [f.filename for f in pdf_files]
    # Update global list of uploaded files for the UI
    temp_uploaded_files_info = [{"name": name} for name in uploaded_file_names]

    try:
        raw_text = extract_text_from_pdfs(pdf_files)
        chunks = chunk_text(raw_text)
        print(f"DEBUG (app.py): Extracted {len(chunks)} text chunks from PDF(s).")
        
        all_triplets = []
        for i, chunk in enumerate(chunks):
            # Call LLM service to extract triplets from each chunk
            triplets = extract_triplets(chunk)
            all_triplets.extend(triplets)
            print(f"DEBUG (app.py): Processed chunk {i+1}/{len(chunks)}, extracted {len(triplets)} triplets from this chunk.")
        
        # Store all extracted triplets in global memory for graph generation step
        temp_extracted_triplets = all_triplets
        print(f"DEBUG (app.py): Total {len(all_triplets)} potential triplets extracted across all chunks.")
        
        return jsonify({
            "status": "success",
            "message": f"Document(s) '{', '.join(uploaded_file_names)}' uploaded and {len(all_triplets)} entities/triplets extracted.",
            "entities_count": len(all_triplets),
            "uploaded_files": temp_uploaded_files_info # Send updated file list back to frontend
        })
    except Exception as e:
        print(f"DEBUG (app.py): Error during PDF processing: {e}")
        return jsonify({"status": "error", "message": f"Error processing PDF: {str(e)}"}), 500

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    """
    Generates the Neo4j graph using the temporarily stored triplets.
    Clears the temporary triplet storage after successful insertion.
    """
    global temp_extracted_triplets

    if not temp_extracted_triplets:
        return jsonify({"status": "warning", "message": "No triplets extracted to generate graph. Please upload PDFs first."})

    try:
        inserted_count = 0
        for i, t in enumerate(temp_extracted_triplets):
            sub = str(t.get('subject', ''))
            pred = str(t.get('predicate', ''))
            obj = str(t.get('object', ''))
            
            if sub and pred and obj:
                # Call Neo4j handler to insert triplet
                insert_triplet_to_neo4j(neo4j_driver, sub, pred, obj)
                inserted_count += 1
            else:
                print(f"DEBUG (app.py): Skipping malformed triplet: {t}")
        
        # Clear temporary triplets after they have been inserted into Neo4j
        temp_extracted_triplets = []
        print(f"DEBUG (app.py): Successfully inserted {inserted_count} triplets into Neo4j.")
        return jsonify({"status": "success", "message": "Neo4j Graph Generated successfully!"})
    except Exception as e:
        print(f"DEBUG (app.py): Error generating graph in Neo4j: {e}")
        return jsonify({"status": "error", "message": f"Error generating graph: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles user chat questions.
    Extracts entities, queries Neo4j for facts, and uses RAG with LLM to answer.
    Chat history is managed solely by the frontend in this setup.
    """
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"status": "error", "message": "No question provided"}), 400

    try:
        print(f"\n--- User Question: {user_question} ---")
        # Call LLM service to extract entities from the question
        identified_elements = extract_entities_from_question(user_question)
        identified_entities = identified_elements.get("entities", [])
        identified_relationships = identified_elements.get("relationships", [])
        print(f"DEBUG (app.py): Identified Entities from question: {identified_entities}")
        print(f"DEBUG (app.py): Identified Relationships from question: {identified_relationships}")

        graph_context = ""
        if identified_entities:
            # Call Neo4j handler to query for relevant facts
            retrieved_facts = query_neo4j_for_facts(neo4j_driver, identified_entities)
            print(f"DEBUG (app.py): Retrieved Facts from Neo4j: {retrieved_facts}")

            if retrieved_facts:
                graph_context = "Facts from Knowledge Graph:\n" + "\n".join(retrieved_facts)
                if identified_relationships:
                    graph_context += f"\nUser also implied relationships: {', '.join(identified_relationships)}"
            else:
                graph_context = f"No direct facts found in the knowledge graph for entities: {', '.join(identified_entities)}."
                print(f"DEBUG (app.py): {graph_context}")
        else:
            graph_context = "No specific entities identified in your question to query the knowledge graph."
            print(f"DEBUG (app.py): {graph_context}")
            
        response_text = ""
        # Only call RAG LLM if meaningful context was found from the graph
        if graph_context and not graph_context.startswith("No specific entities") and not graph_context.startswith("No direct facts"):
            response_text = get_rag_response(graph_context, user_question) # Call LLM service for RAG
        else:
            response_text = "The information is not available in the provided context."
            print(f"DEBUG (app.py): RAG skipped or no context, returning default response: {response_text}")

    except Exception as e:
        print(f"DEBUG (app.py): An error occurred during chat processing: {e}")
        response_text = f"An internal error occurred: {str(e)}. Please try again."

    # Return only the response; frontend manages its own chat history
    return jsonify({"response": response_text})

if __name__ == '__main__':
    # Ensure Neo4j driver connection is closed gracefully on shutdown if needed
    # For simple scripts, it's often handled by the process ending.
    # For persistent apps, consider atexit or Flask's teardown_appcontext.
    try:
        app.run(debug=True, port=5001)  # Set debug=True for development, change to False in production
    finally:
        # Close Neo4j driver connection when app shuts down
        if neo4j_driver:
            neo4j_driver.close()
            print("DEBUG: Neo4j driver closed.")