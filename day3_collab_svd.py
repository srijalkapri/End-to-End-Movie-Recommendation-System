import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load merged ratings+movies file
data_path = Path("data") / "ratings_movies.parquet"  # or .csv

if data_path.suffix == ".parquet":
    ratings_movies = pd.read_parquet(data_path)
else:
    ratings_movies = pd.read_csv(data_path)

print("Merged shape:", ratings_movies.shape)
print(ratings_movies.head())

# 2. Build user–movie rating matrix (rows: users, columns: movies)
user_item_matrix = ratings_movies.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0.0,  # unrated movies become 0
)

print("User–item matrix shape:", user_item_matrix.shape)


# 3. Fit TruncatedSVD on the user–item matrix
n_components = 50  # smaller than number of movies
svd = TruncatedSVD(n_components=n_components, random_state=42)

user_factors = svd.fit_transform(user_item_matrix)
# user_factors: each row = user in latent space

movie_factors = svd.components_.T
# movie_factors: each row = movie in same latent space

print("User factors shape:", user_factors.shape)
print("Movie factors shape:", movie_factors.shape)


# 4. Build mappings between ids and matrix indices
user_ids = user_item_matrix.index.to_list()      # list of userIds in row order
movie_ids = user_item_matrix.columns.to_list()   # list of movieIds in col order

user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

# Also keep a lookup from movieId to title
movieid_to_title = (
    ratings_movies.drop_duplicates("movieId")
    .set_index("movieId")["title"]
    .to_dict()
)


def recommend_for_user(user_id, k=10):
    """
    Recommend top-k movies for a given user_id using SVD factors.
    """
    if user_id not in user_id_to_idx:
        print(f"User {user_id} not found")
        return []

    u_idx = user_id_to_idx[user_id]

    # user vector in latent space
    u_vec = user_factors[u_idx]  # shape: (n_components,)

    # Predicted scores for all movies: dot product user_vec · movie_factors
    scores = movie_factors @ u_vec  # shape: (num_movies,)

    # Mask out movies the user has already rated
    rated_movie_ids = set(
        ratings_movies.loc[ratings_movies["userId"] == user_id, "movieId"].tolist()
    )

    candidates = []
    for m_idx, score in enumerate(scores):
        mid = movie_ids[m_idx]
        if mid in rated_movie_ids:
            continue
        candidates.append((mid, score))

    # Sort by score, highest first
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Take top-k
    top_k = candidates[:k]

    # Convert to readable (title, score)
    results = []
    for mid, score in top_k:
        title = movieid_to_title.get(mid, f"Movie {mid}")
        results.append((mid, title, float(score)))

    return results


if __name__ == "__main__":
    user_example = 1
    recs = recommend_for_user(user_example, k=10)
    print(f"\nTop recommendations for user {user_example}:")
    for mid, title, score in recs:
        print(f"- {title} (movieId={mid}, score={score:.3f})")
