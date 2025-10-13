import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import os
import shutil

def generate_embeddings(csv_file, column_name, output_dir="output_embeddings"):
    """Generates and adds sentence embeddings to a CSV file with optimized performance."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Initialize the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Using GPU for encoding")
        else:
            print("Using CPU for encoding")

        texts = df['text'].tolist()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        df['embedding'] = embeddings
        output_file = os.path.join(output_dir, f"free_{column_name}s.csv")
        df.to_csv(output_file, index=False)
        print(f"Embeddings generated and saved to '{output_file}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "free"

    print("Start talks:", datetime.now().strftime("%H:%M:%S"))
    generate_embeddings("SCRAPED_TALKS.csv", "talk", output_dir)
    print("Start paragraphs:", datetime.now().strftime("%H:%M:%S"))
    generate_embeddings("SCRAPED_PARAGRAPHS.csv", "paragraph", output_dir)
    print("Finish:", datetime.now().strftime("%H:%M:%S"))