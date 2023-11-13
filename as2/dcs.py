#!/usr/bin/env python
# coding: utf-8

# ## Discrete cosine similiarity
# 

# In[46]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import argparse


# In[47]:


# Argument parser setup
parser = argparse.ArgumentParser(description="Locality-Sensitive Hashing for Discrete Cosine Similarity")
parser.add_argument("-b", "--bands", type=int, required=True, help="Number of bands for LSH")
parser.add_argument("-H", "--hashes", type=int, required=True, help="Number of hash functions for Minhash")
args = parser.parse_args()


# Use the arguments in the script
h = args.hashes
b = args.bands

# data load and conversion to coo_matrix format
print(f"Code running for {h} hashes and {b} bands.....")
data = np.load('data/user_movie_rating.npy')
sparse_matrix = coo_matrix(data)

# Save the COO format sparse matrix to a .npz file
save_npz('coo_sparse_matrix.npz', sparse_matrix)

# Print the COO format sparse matrix
print("COO format sparse matrix:")
print(sparse_matrix)


# In[48]:


# get num of users and num of movies

data = np.load('data/user_movie_rating.npy')
data_array = data.astype(int)

# Extract user, movie, and rating data from the loaded records
user_ids, movie_ids, ratings = data[:, 0], data[:, 1], data[:, 2]

# Create a COO (Coordinate List) sparse matrix from the data
user_movie_matrix = coo_matrix((ratings, (user_ids, movie_ids)))


# Load the user-movie ratings data from the npz file
num_users = user_movie_matrix.shape[0]
num_movies = user_movie_matrix.shape[1]

print(num_movies)
print(num_users)


# In[49]:


# Sample function to generate user-movie matrix 
def generate_sample_data(num_users, num_movies):
    data = np.random.randint(2, size=(num_users, num_movies))  # Binary ratings (0 or 1)
    return data


user_movie_matrix = generate_sample_data(num_users, num_movies)

# Print the user-movie matrix (sample)
print("User-Movie Matrix:")
print(user_movie_matrix)


# ## Minhashing

# In[50]:


import numpy as np

# define the number of permutations (h)
h = 100  # You can adjust this number based on your dataset size and desired accuracy

# initialize Minhash signatures for each user
minhash_signatures = np.full((num_users, h), np.inf)

# generate h different random permutations of the columns (movies)
permutations = [np.random.permutation(num_movies) for _ in range(h)]

# compute Minhash signatures for each user
for user_id in range(num_users):
    user_ratings = user_movie_matrix[user_id]  # extract user's ratings (no need for toarray())
    for i in range(h):
        permuted_movie_ids = permutations[i]
        permuted_ratings = user_ratings[permuted_movie_ids]
        first_nonzero_index = np.argmax(permuted_ratings)
        minhash_signatures[user_id, i] = permuted_movie_ids[first_nonzero_index]


# In[51]:


print("Number of Minhash Signatures:", minhash_signatures.shape[1])


# In[52]:


#  LSH with Minhash Signatures

#  number of bands (b) and rows per band (r)
b = 10  # number of partitions
r = h // b  # ensure that r * b = h

# initialize a dictionary to store buckets
buckets = {}

# hash Minhash signatures into bands
for user_id in range(num_users):
    minhash_signature = minhash_signatures[user_id]
    for band_id in range(b):
        band_signature = minhash_signature[band_id * r : (band_id + 1) * r]
        # convert the band signature to a hashable string using hash fnc
        band_signature_str = str(band_signature)
        # add the user to the corresponding bucket
        if band_signature_str not in buckets:
            buckets[band_signature_str] = []
        buckets[band_signature_str].append(user_id)



# In[53]:


# Pair Generation

# initialize a list to store candidate pairs of users
candidate_pairs = []

# iterate through the buckets created in Step 3
for bucket in buckets.values():
    # generate pairs of users within each bucket
    for i in range(len(bucket)):
        for j in range(i + 1, len(bucket)):
            user1 = bucket[i]
            user2 = bucket[j]
            candidate_pairs.append((user1, user2))


# In[54]:


#  DCS Calculation and Threshold Check

# define the threshold for DCS
threshold_dcs = 0.73  

# initialize a list to store pairs of users with high DCS
similar_user_pairs = []

# calculate DCS between two Minhash signatures
def calculate_dcs(minhash_signature1, minhash_signature2):
    # Replace every non-zero rating with 1 in both signatures
    minhash_signature1 = (minhash_signature1 > 0).astype(int)
    minhash_signature2 = (minhash_signature2 > 0).astype(int)

    # compute the cosine similarity between the modified signatures
    dot_product = np.dot(minhash_signature1, minhash_signature2)
    norm1 = np.linalg.norm(minhash_signature1)
    norm2 = np.linalg.norm(minhash_signature2)

    # calculate the DCS value
    dcs = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0

    return dcs
    
# iterate through the candidate pairs
for user1, user2 in candidate_pairs:
    # retrieve the Minhash signatures for user1 and user2
    minhash_signature1 = minhash_signatures[user1]
    minhash_signature2 = minhash_signatures[user2]
    
    # calculate DCS between the Minhash signatures
    dcs = calculate_dcs(minhash_signature1, minhash_signature2)
    
    # check if DCS exceeds the threshold
    if dcs > threshold_dcs:
        similar_user_pairs.append((user1, user2))


# In[55]:


# Output

# Define the output file name
output_file = "similar_user_pairs.txt"

# write similar user pairs to the output file
with open(output_file, "w") as file:
    for user1, user2 in similar_user_pairs:
        # write the user pair (u1, u2) to the file
        file.write(f"{user1},{user2}\n")
