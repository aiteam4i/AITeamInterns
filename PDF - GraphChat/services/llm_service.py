import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Ensure .env is loaded. This is good practice if this module might be run or tested standalone.
load_dotenv() 

# --- Configuration for Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# Define the Groq models to be used
# You can choose between "llama-3.1-8b-instant" for faster/cheaper, or "llama-3.1-70b-versatile" for more robust.
GROQ_MODEL_GRAPH_EXTRACTION = "llama-3.3-70b-versatile" 
GROQ_MODEL_RAG = "llama-3.3-70b-versatile"

def extract_triplets(text_chunk):
    """
    Uses the LLM to extract subject-predicate-object triplets from a text chunk.
    """
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

    content = ""
    try:
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL_GRAPH_EXTRACTION,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Lower temperature for more deterministic triplet extraction
            response_format={"type": "json_object"} # Request JSON output
        )
        content = chat_completion.choices[0].message.content
        
        parsed_response = json.loads(content)

        # Handle cases where LLM might wrap the array in an object (e.g., {"triplets": [...]})
        if isinstance(parsed_response, dict) and "triplets" in parsed_response and isinstance(parsed_response["triplets"], list):
            return parsed_response["triplets"]
        elif isinstance(parsed_response, list):
            return parsed_response
        else:
            print(f"DEBUG (llm_service.py): LLM returned unexpected root format for triplets after JSON parse: {content}")
            return []
    except json.JSONDecodeError as e:
        print(f"DEBUG (llm_service.py): JSON parse error in extract_triplets: {e}. Raw content: {content[:200]}...")
        return []
    except Exception as e:
        print(f"DEBUG (llm_service.py): Error in extract_triplets: {e}")
        return []

def extract_entities_from_question(question):
    """
    Uses the LLM to identify key entities and relationships from a user's question.
    """
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

    response_content = ""
    try:
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL_GRAPH_EXTRACTION, # Using the same model for consistency, can be different
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Lower temperature for more deterministic entity extraction
            response_format={"type": "json_object"} # Request JSON output
        )
        response_content = chat_completion.choices[0].message.content
        
        parsed_response = json.loads(response_content)

        if isinstance(parsed_response, dict) and "entities" in parsed_response and "relationships" in parsed_response:
            return parsed_response
        else:
            print(f"DEBUG (llm_service.py): LLM returned unexpected format for entities/relationships after JSON parse: {response_content}")
            return {"entities": [], "relationships": []}

    except json.JSONDecodeError as e:
        print(f"DEBUG (llm_service.py): JSON decoding error from LLM response for entities: {e}. Raw output: {response_content[:500]}...")
        return {"entities": [], "relationships": []}
    except Exception as e:
        print(f"DEBUG (llm_service.py): Error calling LLM for entity extraction: {e}. Raw LLM output: {response_content[:500]}...")
        return {"entities": [], "relationships": []}

def get_rag_response(graph_context, user_question):
    """
    Generates a RAG-based answer using the retrieved graph context and user question.
    """
    rag_prompt_str = f"""
You are a helpful assistant specialized in providing answers based on the provided context.
Answer the user's question truthfully and concisely based  on the following facts.
If the answer cannot be found in the provided facts, state that the information is not available in the provided context.

Facts from Knowledge Graph:
{graph_context}

Question:
{user_question}

Answer:
"""
    print(f"DEBUG (llm_service.py): RAG Prompt sent to LLM:\n{rag_prompt_str}")
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_RAG,
            messages=[{"role": "user", "content": rag_prompt_str}],
            temperature=0.5, # Slightly higher temperature for more natural language generation
        )
        response_text = response.choices[0].message.content
        print(f"DEBUG (llm_service.py): LLM RAG Response: {response_text}")
        return response_text
    except Exception as e:
        print(f"DEBUG (llm_service.py): Error getting RAG response from LLM: {e}")
        return "An error occurred while generating a response."