import faiss
import torch
import numpy as np
from line_profiler import LineProfiler

def find_closest_embedding(input_embedding, angle, index, input_index, exclude_indices, max_results=1):
    min_results = 5  # Minimum number of results to consider before filtering

    # Calculate the cosine of the angle
    cosine_angle = np.cos(np.radians(angle))

    # Reshape and normalize the input embedding
    input_embedding = input_embedding.reshape(1, -1)
    input_norm = np.linalg.norm(input_embedding)
    normalized_input_embedding = input_embedding / input_norm

    # Add the input_index to the exclude_indices
    exclude_indices.add(input_index)

    # Search for a larger set of results
    d, idx = index.search(cosine_angle * normalized_input_embedding, max(min_results, max_results))

    # Filter out the excluded indices
    filtered_indices = [i for i in idx[0] if i not in exclude_indices]

    # Take the closest embeddings after filtering
    closest_embeddings = filtered_indices[:max_results]

    return closest_embeddings

# Setup a dummy FAISS index and data for testing
dimension = 128
faiss_index = faiss.IndexFlatIP(dimension)
data = np.random.random((100, dimension)).astype('float32')
faiss_index.add(data)

# Random input embedding
input_embedding = np.random.random(dimension).astype('float32')
angle = 45
input_index = 10
exclude_indices = set()

# Profiling the function
profiler = LineProfiler()
profiler.add_function(find_closest_embedding)
profiler.enable_by_count()

closest_embeddings = find_closest_embedding(input_embedding, angle, faiss_index, input_index, exclude_indices, max_results=1)

profiler.disable_by_count()
profiler.print_stats()
