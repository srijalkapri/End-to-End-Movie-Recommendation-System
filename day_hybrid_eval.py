import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.decomposition import TruncatedSVD


data_path = Path("data") / "ratings_movies.parquet"   # or .csv

if data_path.suffix == ".parquet":
    ratings_movies = pd.read_parquet(data_path)
else:
    ratings_movies = pd.read_csv(data_path)

print("Data shape:", ratings_movies.shape)
print(ratings_movies.head())



def train_test_split_by_user(df, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    train_rows = []
    test_rows = []

    for user_id, group in df.groupby("userId"):
        idx = np.arange(len(group))
        np.random.shuffle(idx)

        cutoff = int(len(group) * (1 - test_size))
        train_idx = idx[:cutoff]
        test_idx = idx[cutoff:]

        train_rows.append(group.iloc[train_idx])
        test_rows.append(group.iloc[test_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


train_df, test_df = train_test_split_by_user(ratings_movies, test_size=0.2)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)





# 3. Build user–movie matrix on TRAIN only
train_user_item = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0.0,
)

print("Train user–item shape:", train_user_item.shape)

# Fit SVD on train matrix
n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(train_user_item)
movie_factors = svd.components_.T

# Id mappings based on TRAIN users/movies
user_ids = train_user_item.index.to_list()
movie_ids = train_user_item.columns.to_list()
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}


def predict_rating(user_id, movie_id):
    # If user or movie not seen in train, return a default (e.g., global mean)
    if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx:
        return train_df["rating"].mean()

    u_idx = user_id_to_idx[user_id]
    m_idx = movie_id_to_idx[movie_id]

    u_vec = user_factors[u_idx]
    m_vec = movie_factors[m_idx]

    # Dot product gives predicted rating
    pred = float(np.dot(u_vec, m_vec))
    return pred


from sklearn.metrics import mean_squared_error

y_true = []
y_pred = []

for _, row in test_df.iterrows():
    uid = row["userId"]
    mid = row["movieId"]
    true_r = row["rating"]
    pred_r = predict_rating(uid, mid)

    y_true.append(true_r)
    y_pred.append(pred_r)

mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5

print(f"\nTest RMSE (collaborative SVD): {rmse:.4f}")



# Build movieId -> title map
movieid_to_title = (
    ratings_movies.drop_duplicates("movieId")
    .set_index("movieId")["title"]
    .to_dict()
)

def recommend_for_user_svd(user_id, k=10):
    if user_id not in user_id_to_idx:
        return []

    u_idx = user_id_to_idx[user_id]
    u_vec = user_factors[u_idx]
    scores = movie_factors @ u_vec

    # movies already in TRAIN for this user
    rated_in_train = set(
        train_df.loc[train_df["userId"] == user_id, "movieId"].tolist()
    )

    candidates = []
    for m_idx, score in enumerate(scores):
        mid = movie_ids[m_idx]
        if mid in rated_in_train:
            continue
        candidates.append((mid, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = candidates[:k]
    return [mid for mid, _ in top_k]



def evaluate_topk_hit_precision(test_df, k=10):
    hits = 0
    total_users = 0
    precisions = []

    for user_id, group in test_df.groupby("userId"):
        # movies this user rated in TEST = "relevant"
        relevant_movies = set(group["movieId"].tolist())
        if not relevant_movies:
            continue

        recs = recommend_for_user_svd(user_id, k=k)
        if not recs:
            continue

        total_users += 1

        # Hit@K: at least one test movie in recommendations
        if any(mid in relevant_movies for mid in recs):
            hits += 1

        # Precision@K: fraction of recs that are in test set
        num_relevant_in_recs = sum(1 for mid in recs if mid in relevant_movies)
        precisions.append(num_relevant_in_recs / k)

    hit_at_k = hits / total_users if total_users > 0 else 0.0
    precision_at_k = sum(precisions) / len(precisions) if precisions else 0.0
    return hit_at_k, precision_at_k


hit10, prec10 = evaluate_topk_hit_precision(test_df, k=10)
print(f"Hit@10: {hit10:.4f}")
print(f"Precision@10: {prec10:.4f}")



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Build a small movies table
movies = ratings_movies.drop_duplicates("movieId")[["movieId", "title", "genres"]]

def clean_genres(genres_str):
    if pd.isna(genres_str):
        return ""
    return genres_str.replace("|", " ").lower()

movies["content"] = movies["genres"].apply(clean_genres)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["content"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movieid_to_idx = {mid: i for i, mid in enumerate(movies["movieId"].tolist())}
idx_to_movieid = {i: mid for mid, i in movieid_to_idx.items()}



def hybrid_recommend_for_user(user_id, k=10, alpha=0.7):
    """
    alpha: weight for SVD score (0-1). (1-alpha) for content.
    """
    if user_id not in user_id_to_idx:
        return []

    u_idx = user_id_to_idx[user_id]
    u_vec = user_factors[u_idx]
    svd_scores = movie_factors @ u_vec  # one score per movie index

    # Movies user liked (rating >= 4.0) in TRAIN
    liked = train_df[(train_df["userId"] == user_id) & (train_df["rating"] >= 4.0)]
    liked_movie_ids = liked["movieId"].tolist()
    liked_indices = [movieid_to_idx[mid] for mid in liked_movie_ids if mid in movieid_to_idx]

    candidates = []
    for m_idx, svd_score in enumerate(svd_scores):
        mid = movie_ids[m_idx]

        # skip movies in TRAIN (already seen)
        if mid in liked_movie_ids:
            continue

        # content score: max similarity to any liked movie
        if liked_indices:
            sim_vec = cosine_sim[m_idx, liked_indices]
            content_score = float(sim_vec.max())
        else:
            content_score = 0.0

        hybrid_score = alpha * svd_score + (1 - alpha) * content_score
        candidates.append((mid, hybrid_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = candidates[:k]

    results = []
    for mid, score in top_k:
        title = movieid_to_title.get(mid, f"Movie {mid}")
        results.append((mid, title, float(score)))
    return results

if __name__ == "__main__":
    user_example = 1
    recs = hybrid_recommend_for_user(user_example, k=10)
    print(f"\nHybrid recommendations for user {user_example}:")
    for mid, title, score in recs:
        print(f"- {title} (movieId={mid}, score={score:.3f})")


