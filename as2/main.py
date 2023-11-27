import argparse
import numpy as np
import itertools
from scipy.sparse import csr_matrix
import time

def hash_function(input, coef1, coef2, coef3, n_buckets):
    return (coef1*input**2 + coef2*input + coef3) % n_buckets

def minhash(user_movie_matrix, num_hashes):

    num_users = user_movie_matrix.shape[0]
    num_movies = user_movie_matrix.shape[1]
    # generate hash functions by representing them as the coefficients 
    hash_functions = np.random.randint(1,1000,(num_hashes,3))
    
    # generate signature matrix
    signature_matrix = np.full((num_hashes,num_users),np.inf)

    # Update signature matrix for each user in the rating matrix using the hash functions
    # Start with user 1 and iterate over all users
    for user in range(num_users):
        # Find indices of movies that the user has rated
        non_zero_indices = user_movie_matrix[user].nonzero()[1]

        if len(non_zero_indices) == 0:
            continue

        # Calculate hash value for each hash function for each movie that the user has rated and keep the minimum hash value
        minhash_values = np.array([min(hash_function(non_zero_indices,hash_functions[i][0],hash_functions[i][1],hash_functions[i][2],num_movies)) for i in range(num_hashes)])
        # Update signature matrix
        signature_matrix[:,user] = np.minimum(minhash_values, signature_matrix[:,user])

    return signature_matrix
def lsh(signature_matrix , bands, similarity_function, threshold):
    print("LSH input:", signature_matrix)  # Debug step 0
    num_hashes, num_users = signature_matrix.shape
    rows_per_band = num_hashes // bands    

    # Generate hash functions for each band
    hash_functions = np.random.randint(1,1000,(bands,3))
    n_buckets = num_users // 2

    # Initialize a dictionary to store candidate pairs
    candidate_pairs = {}

    # Apply LSH to find candidate pairs
    for band in range(bands):
        # Extract a band from the signature matrix
        band_matrix = signature_matrix[band * rows_per_band: (band + 1) * rows_per_band, :]

        # Collapse the rows of the band into a single row for each user
        band_value = np.sum(band_matrix, axis=0)

        # Calculate destination bucket for each user in the band
        hash_values = hash_function(band_value, hash_functions[band][0], hash_functions[band][1], hash_functions[band][2], n_buckets)

        # Map each user to its bucket
        for user, hash_value in enumerate(hash_values):
            # Add the current user to the candidate list
            candidate_pairs.setdefault(hash_value, []).append(user)

    # Find candidate pairs
    similar_users = set()
    for bucket in candidate_pairs.values():
        # Generate all possible pairs of users in the bucket
        pairs = itertools.combinations(bucket, 2)

        # Calculate the similarity of each pair
        for user1, user2 in pairs:
            similarity = similarity_function(signature_matrix[:,user1], signature_matrix[:,user2])

            # If the similarity is above the threshold, add the pair to the candidate list
            if similarity > threshold:
                similar_users.add((user1, user2))
    print("LSH output:", similar_users)  # Debug step
    return similar_users

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


# def test_jaccard_similarity():
#     # Define two sets
#     set1 = set([1, 2, 3, 4])
#     set2 = set([3, 4, 5, 6])

#     # Calculate expected Jaccard similarity
#     expected_similarity = len(set1.intersection(set2)) / len(set1.union(set2))

#     # Call the function
#     actual_similarity = jaccard_similarity(set1, set2)

#     # Check if the output matches the expected similarity
#     assert actual_similarity == expected_similarity, f"Expected {expected_similarity}, but got {actual_similarity}"


def main():
    # test_jaccard_similarity()
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Calculate Jaccard similarity.')
    parser.add_argument('-d', type=str, help='Data file path')
    parser.add_argument('-s', type=int, help='Random seed')
    parser.add_argument('-m', type=str, help='Similarity measure')

    args = parser.parse_args()

    np.random.seed(args.s)

    if args.m == 'js':
        # Load data from file
        data = np.load(args.d)
        user_ids, movie_ids, ratings = data[:, 0], data[:, 1], data[:, 2]
        user_movie_matrix = csr_matrix((ratings, (user_ids, movie_ids)))
        num_users = user_movie_matrix.shape[0]
        num_movies = user_movie_matrix.shape[1]
        # Calculate Jaccard similarity and write results to js.txt
        # with open('js.txt', 'w') as f:
        #     num_hashes = 100
        #     signature_matrix = minhash(user_movie_matrix,num_hashes)
        #     similar_users = lsh(signature_matrix, bands=14, similarity_function=jaccard_similarity, threshold=0.5)
        #     for user1, user2 in similar_users:
        #         f.write(f'{user1}, {user2}\n')
        
        with open('js.txt', 'w') as f:
            num_hashes = 100
            signature_matrix = minhash(user_movie_matrix,num_hashes)
            print(signature_matrix)  # Debug step 1
            similar_users = lsh(signature_matrix, bands=14, similarity_function=jaccard_similarity, threshold=0.5)
            print(similar_users)  # Debug step 2
            for user1, user2 in similar_users:
                f.write(f'{user1}, {user2}\n')
        

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()