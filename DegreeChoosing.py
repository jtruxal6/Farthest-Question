import faiss
import torch
import numpy as np
from embedding_utils import find_closest_embedding

# Load embeddings from a file
embeddings = torch.load('./train_embeddings.pt')
print("done loading")

# Convert to the required format
pooler_outputs = [i["embedding"]["pooler_output"] for i in embeddings]

# Select an embedding as the input - for example, the first embedding in the list
selectedEmbedding = 0
input_embedding = pooler_outputs[selectedEmbedding]

# Print the inputted question
input_embedding_data = embeddings[selectedEmbedding]
input_videoID = list(input_embedding_data.keys())[0]
input_question = input_embedding_data[input_videoID]
print("Inputted question:", input_question)

# Loop through angles from 0 to 180 in increments of 5
for angle in range(0, 181, 5):
    # Find the closest embeddings excluding the input embedding
    closest_idx = find_closest_embedding(input_embedding, angle, pooler_outputs, selectedEmbedding, max_results=1)[0]

    # Access the data for this embedding
    closest_embedding_data = embeddings[closest_idx]

    # Extract question and video ID
    videoID = list(closest_embedding_data.keys())[0]
    question = closest_embedding_data[videoID]

    # Print the angle and the corresponding question
    print("Angle:", angle, "- Closest question:", question)

# You can now use closest_embeddings as needed
import IPython; IPython.embed()