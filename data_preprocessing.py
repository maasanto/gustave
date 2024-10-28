import os
import re
import json

def extract_qa_pairs_from_text(text):
    """
    Extract question-answer pairs from documentation text based on patterns.
    Assumes lines that start with 'Q:' are questions and 'A:' are answers.
    """
    qa_pairs = []
    lines = text.splitlines()
    question, answer = None, []
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            # Save the previous question-answer pair if it exists
            if question and answer:
                qa_pairs.append({"question": question, "answer": ' '.join(answer)})
            question = line[2:].strip()
            answer = []
        elif line.startswith("A:"):
            answer.append(line[2:].strip())
        elif answer:
            answer.append(line)  # Continuation of the answer text
    
    # Append the last Q&A pair if any
    if question and answer:
        qa_pairs.append({"question": question, "answer": ' '.join(answer)})

    return qa_pairs

def process_file(file_path):
    """
    Process a single file to extract Q&A pairs.
    Currently supports Markdown-style Q&A and comments in code files.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    qa_pairs = extract_qa_pairs_from_text(content)
    return qa_pairs

def scan_directory_for_qa_pairs(root_dir, file_extensions=('.md', '.txt', '.py')):
    """
    Traverse through directories, extracting Q&A pairs from supported files.
    """
    all_qa_pairs = []
    for dirpath, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith(file_extensions):
                file_path = os.path.join(dirpath, file_name)
                qa_pairs = process_file(file_path)
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
    root_directory = "path/to/your/documentation"  # replace with your directory
    output_file = "qa_pairs.json"

    # Scan the directory and extract Q&A pairs
    qa_pairs = scan_directory_for_qa_pairs(root_directory)

    # Save the results to a JSON file
    save_qa_pairs_to_json(qa_pairs, output_file)
