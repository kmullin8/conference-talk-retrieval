from openai import OpenAI
import pandas as pd
import tiktoken
from datetime import datetime
import json
import os

with open("config.json") as config:
    openaiKey = json.load(config)["openaiKey"]

OpenAI.api_key = openaiKey
client = OpenAI(api_key=OpenAI.api_key)

def get_embedding(texts, output_dir, model="text-embedding-3-small", max_tokens=300000):
    """
    Generate embeddings for a list of texts in batches, respecting token limits.
    
    Args:
        texts: List of strings to embed
        model: Embedding model name
        max_tokens: Maximum tokens per API request
    
    Returns:
        List of embeddings
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer
    encoder = tiktoken.encoding_for_model(model)
    
    # Clean texts and calculate token counts
    texts = [text.replace("\n", " ") for text in texts]
    token_counts = [len(encoder.encode(text)) for text in texts]
    
    embeddings = []
    current_batch = []
    current_token_count = 0
    
    for i, (text, token_count) in enumerate(zip(texts, token_counts)):
        if current_token_count + token_count > max_tokens or len(current_batch) >= 100:
            # Process current batch
            response = client.embeddings.create(input=current_batch, model=model)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            # Reset batch
            current_batch = [text]
            current_token_count = token_count
        else:
            current_batch.append(text)
            current_token_count += token_count
    
    # Process final batch
    if current_batch:
        response = client.embeddings.create(input=current_batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return embeddings

if __name__ == "__main__":
    output_dir = "openai"

    # Process talks.csv
    df = pd.read_csv("SCRAPED_TALKS.csv")
    df['embedding'] = get_embedding(df['text'].tolist(), output_dir, model='text-embedding-3-small')
    output_talks = os.path.join(output_dir, 'openai_talks.csv')
    df.to_csv(output_talks, index=False)

    # Process paragraphs.csv
    df = pd.read_csv("SCRAPED_PARAGRAPHS.csv")
    df['embedding'] = get_embedding(df['text'].tolist(), output_dir, model='text-embedding-3-small')
    output_paragraphs = os.path.join(output_dir, 'openai_paragraphs.csv')
    df.to_csv(output_paragraphs, index=False)

    file_to_delete = "SCRAPED_TALKS.csv"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"Deleted file: {file_to_delete}")
    file_to_delete = "SCRAPED_PARAGRAPHS.csv"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"Deleted file: {file_to_delete}")

    