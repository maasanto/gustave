import os
import re
import json
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.core.node_parser import SimpleNodeParser

# Set your API key here, or load from an environment variable for security
# openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_sections(content, file_extension):
    """
    Extracts relevant sections from text, targeting comments or sections in 
    documentation and code files based on file type.
    """
    if file_extension in {".md", ".txt"}:
        # Example: Split Markdown or text documentation by headers
        sections = re.split(r'(^#{1,6} .+)', content, flags=re.MULTILINE)
    elif file_extension == ".py":
        # Extract docstrings and comments for Python files
        sections = re.findall(r'(?:\'\'\'|\"\"\")([\s\S]*?)(?:\'\'\'|\"\"\")|(#.*)', content)
        sections = [s[0] or s[1] for s in sections]  # Flatten the tuples
    else:
        sections = [content]  # Fallback for other file types
    return sections

def generate_qa_pairs_with_llm(section_text):
    """
    Uses an LLM with llama-index to generate Q&A pairs for a given section of text.
    """
    
	# Use local embedding model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embedding_name = "OrdalieTech/Solon-embeddings-base-0.1"
    embed_model = HuggingFaceEmbedding(model_name=embedding_name)
    Settings.embed_model = embed_model

    # Set up the LLM settings
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create a Document object from the section text
    document = Document(text=section_text)

    # Create a node parser
    node_parser = SimpleNodeParser()

    # Get nodes from the document
    nodes = node_parser.get_nodes_from_documents([document])  # Pass the document as a list
	
	# Build the index using llama-index
    index = VectorStoreIndex(nodes)

    # Query the index to generate Q&A
    prompt = "Generate multiple question and answer pairs based on the provided content that a user of the support would ask on a forum. Use the following format QUESTION: content/nANSWER: content"
    
	# Use the LLM to run the query
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    # Parse response into a dictionary for Q&A
    qa_pairs = response.response.split('\n')
    question = qa_pairs[0].replace("QUESTION:", " ").strip()
    answer = qa_pairs[1].replace("ANSWER:", " ").strip()

    return {"question": question, "answer": answer}

def process_file_with_llm(file_path):
    """
    Process a file to extract sections and use the LLM to create Q&A pairs.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract sections
    file_extension = os.path.splitext(file_path)[1]
    sections = extract_text_sections(content, file_extension)

    # Generate Q&A pairs for each section
    qa_pairs = []
    for section in sections:
        if section.strip():  # Avoid empty sections
            qa_pair = generate_qa_pairs_with_llm(section.strip())
            qa_pairs.append(qa_pair)

    return qa_pairs

def scan_directory_with_llm(root_dir, file_extensions=('.md', '.txt', '.py')):
    """
    Traverse directories, using LLM to generate Q&A pairs from supported files.
    """
    all_qa_pairs = []
    for dirpath, _, files in os.walk(root_dir):
        for file_name in files:
            print(file_name)
            if file_name.endswith(file_extensions):
                file_path = os.path.join(dirpath, file_name)
                qa_pairs = process_file_with_llm(file_path)
                all_qa_pairs.extend(qa_pairs)
    return all_qa_pairs

def save_qa_pairs_to_json(qa_pairs, output_file):
    """
    Save extracted Q&A pairs to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Q&A pairs saved to {output_file}")

if __name__ == "__main__":
    # Set the directory to scan and output file name
    root_directory = "/Users/tat/dev-local/data_sample"  # replace with your directory
    output_file = "qa_pairs.json"

    # Scan the directory and extract Q&A pairs
    qa_pairs = scan_directory_with_llm(root_directory)
    print(qa_pairs)
    # Save the results to a JSON file
    save_qa_pairs_to_json(qa_pairs, output_file)
