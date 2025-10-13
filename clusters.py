import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import ast
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cluster_paragraph_embeddings(csv_file, k, prefix):
    """
    Generate k cluster embeddings per talk from paragraph embeddings by clustering paragraphs
    within each talk and using cluster centroids as the new embeddings.
    
    Parameters:
    - csv_file: Path to CSV file containing paragraph embeddings
    - k: Number of clusters (default=3)
    
    Returns:
    - DataFrame containing cluster embeddings and metadata
    """
    try:
        # Load the paragraph embeddings CSV
        open_file = os.path.join(prefix, csv_file)
        df = pd.read_csv(open_file)
        logging.info(f"Loaded CSV with {len(df)} paragraphs")

        # Validate required columns
        required_columns = ['url', 'embedding', 'title', 'speaker', 'calling', 'year', 'season']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # Convert string embeddings to numpy arrays
        df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
        
        # Group paragraphs by talk (using url as unique identifier for talks)
        grouped = df.groupby('url')
        cluster_data = []
        
        for talk_url, group in grouped:
            # Extract metadata for the talk
            talk_info = group.iloc[0][['title', 'speaker', 'calling', 'year', 'season', 'url']].to_dict()
            
            # Get all embeddings for the talk
            embeddings = np.stack(group['embedding'].values)
            paragraph_texts = group['text'].values
            
            # Check if there are enough paragraphs for clustering
            if len(embeddings) < k:
                logging.warning(f"Talk {talk_info['title']} ({talk_url}) has {len(embeddings)} "
                              f"paragraphs, fewer than k={k}. Skipping clustering.")
                continue
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            
            # Get cluster centroids (these are the new embeddings)
            centroids = kmeans.cluster_centers_
            
            # Store each centroid as a new embedding for the talk
            for cluster_idx in range(k):
                cluster_centroid = centroids[cluster_idx]
                similarities = cosine_similarity([cluster_centroid], embeddings)[0]
                top_indices = np.argsort(similarities)[-3:]
                top_indices = list(reversed(top_indices))  # Most similar first
                top_paragraphs = [paragraph_texts[i] for i in top_indices]

                cluster_data.append({
                    'title': talk_info['title'],
                    'speaker': talk_info['speaker'],
                    'calling': talk_info['calling'],
                    'year': talk_info['year'],
                    'season': talk_info['season'],
                    'url': talk_info['url'],
                    'cluster_id': cluster_idx + 1,
                    'text': top_paragraphs,
                    'embedding': cluster_centroid.tolist()
                })
            
            # logging.info(f"Generated {k} cluster embeddings for talk: {talk_info['title']} ({talk_url})")
        
        # Create DataFrame for cluster embeddings
        if not cluster_data:
            raise ValueError("No cluster embeddings generated. Check input data or clustering process.")
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Save to CSV
        output_file = prefix + '_' + str(k) + '_clusters.csv'
        cluster_df.to_csv(os.path.join(prefix, output_file), index=False)
        logging.info(f"Cluster embeddings saved to {output_file}")
        
        return cluster_df
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

# Execute the clustering
if __name__ == "__main__":
    print("Start paragraphs:", datetime.now().strftime("%H:%M:%S"))
    cluster_paragraph_embeddings("free_paragraphs.csv", 3, "free")
    cluster_paragraph_embeddings("openai_paragraphs.csv", 3, "openai")
    print("Finish:", datetime.now().strftime("%H:%M:%S"))