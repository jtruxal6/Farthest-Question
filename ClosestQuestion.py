import faiss
import torch
import numpy as np
from embedding_utils import find_closest_embedding
from tqdm import tqdm
import random

def load_embeddings(file_path, max_videos=None):
    embeddings = torch.load(file_path)
    embeddings_by_videoID = {}

    if max_videos is not None:
        all_videoIDs = {"_".join(full_videoID.split("_")[:-1]) for emb_data in embeddings for full_videoID in emb_data}
        selected_videoIDs = set(random.sample(all_videoIDs, min(max_videos, len(all_videoIDs))))
    else:
        selected_videoIDs = None

    for embedding_data in embeddings:
        for full_videoID, question_text in embedding_data.items():
            base_videoID = "_".join(full_videoID.split("_")[:-1])
            if selected_videoIDs is None or base_videoID in selected_videoIDs:
                embeddings_by_videoID.setdefault(base_videoID, []).append((embedding_data['embedding']['pooler_output'].numpy(), question_text))

    return embeddings_by_videoID

def prepare_embeddings(embeddings_by_videoID):
    all_embeddings = []
    all_questions = []
    videoID_to_indices = {}

    for videoID, video_embeddings in embeddings_by_videoID.items():
        start_index = len(all_embeddings)
        all_embeddings.extend([emb for emb, _ in video_embeddings])
        all_questions.extend([ques for _, ques in video_embeddings])
        videoID_to_indices[videoID] = list(range(start_index, len(all_embeddings)))

    return all_embeddings, all_questions, videoID_to_indices


def find_closest_embeddings(embeddings_by_videoID, all_embeddings, all_questions, videoID_to_indices, angle=95):
    exclusion_sets = {videoID: set(indices) for videoID, indices in videoID_to_indices.items()}
    closest_embeddings_questions_by_videoID = {}

    # Convert embeddings to numpy array if they are in torch format
    if isinstance(all_embeddings[0], torch.Tensor):
        all_embeddings = torch.stack(all_embeddings).numpy().astype("float32")

    # Normalize the embeddings
    vector_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    normalized_embeddings = all_embeddings / vector_norms

    # Create FAISS index
    passInindex = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    passInindex.add(normalized_embeddings)

    for videoID, indices in tqdm(videoID_to_indices.items(), desc="Processing videos"):
        closest_embeddings_questions = []
        for index in indices:
            input_embedding = all_embeddings[index]
            exclude_indices = exclusion_sets[videoID] - {index}
            closest_idx = find_closest_embedding(input_embedding, angle, passInindex, index, exclude_indices, max_results=1)
            
            if closest_idx and closest_idx[0] != -1:
                closest_embeddings_questions.append((all_embeddings[closest_idx[0]], all_questions[closest_idx[0]]))

        closest_embeddings_questions_by_videoID[videoID] = closest_embeddings_questions

    return closest_embeddings_questions_by_videoID



def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Parameters for testing
max_videos_for_testing = None # Adjust as needed for testing

# Load and prepare a subset of embeddings
embeddings_by_videoID = load_embeddings('./train_embeddings.pt', max_videos=max_videos_for_testing)
all_embeddings, all_questions, videoID_to_indices = prepare_embeddings(embeddings_by_videoID)


# Find closest embeddings and questions on the subset
closest_embeddings_questions_by_videoID = find_closest_embeddings(embeddings_by_videoID, all_embeddings, all_questions, videoID_to_indices)

# Process and select the closest question for each video
final_closest_question_by_videoID = {}

for videoID, closest_embeddings_questions in closest_embeddings_questions_by_videoID.items():
    video_embeddings = [emb for emb, _ in embeddings_by_videoID[videoID]]  # Extract only embeddings
    video_questions = [ques for _, ques in embeddings_by_videoID[videoID]]  # Extract only questions

    min_avg_distance = float('inf')
    selected_closest_question_text = None

    for closest_embedding, closest_question in closest_embeddings_questions:
        total_distance = sum(calculate_distance(original_embedding, closest_embedding) for original_embedding in video_embeddings)
        avg_distance = total_distance / len(video_embeddings)

        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            selected_closest_question_text = closest_question

    final_closest_question_by_videoID[videoID] = selected_closest_question_text

# Display each original question alongside the selected closest question for each video
for videoID in final_closest_question_by_videoID:
    print(f"\nVideo ID: {videoID}")
    video_questions = [ques for _, ques in embeddings_by_videoID[videoID]]  # Extract only questions for the current video
    print("Original Questions and Selected Closest Question:")
    for question in video_questions:
        print(f"  Original: {question}")
    print(f"  Selected Closest: {final_closest_question_by_videoID[videoID]}")