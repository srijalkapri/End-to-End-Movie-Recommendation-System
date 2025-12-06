import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Point to your data folder (same as yesterday)
data_dir = Path("C:\\Users\\Srijal\\Downloads\\ml-latest-small")
movies_path = data_dir / "movies.csv"

# 2. Load movies table
movies = pd.read_csv(movies_path)

print("Movies shape:", movies.shape)
print(movies.head())

# 3. Create a clean text column from genres
def clean_genres(genres_str):
    # Handle missing values
    if pd.isna(genres_str):
        return ""
    # Replace "|" with space, make lowercase
    return genres_str.replace("|", " ").lower()

movies["content"] = movies["genres"].apply(clean_genres)

print("\nMovies with content column:")
print(movies[["movieId", "title", "genres", "content"]].head())

# 4. Build TF-IDF matrix from the content column
tfidf = TfidfVectorizer(stop_words="english")  # ignore common English words
tfidf_matrix = tfidf.fit_transform(movies["content"])

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)
# rows = number of movies, columns = number of unique terms


# 5. Compute cosine similarity between all movie vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# cosine_sim[i, j] will be the similarity between movie i and movie j

# Build a mapping from movie title to row index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def get_similar_movies(title, k=10):
    """
    Given a movie title, return top k similar movie titles.
    """
    # 1. Get the index (row number) of this movie
    if title not in indices:
        print(f"Title '{title}' not found!")
        return []

    idx = indices[title]

    # 2. Get similarity scores for this movie to all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 3. Sort movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 4. Take top k+1 (because first one will be the movie itself)
    sim_scores = sim_scores[1 : k + 1]

    # 5. Get the movie indices
    movie_indices = [i for i, score in sim_scores]

    # 6. Return titles of similar movies
    return movies["title"].iloc[movie_indices].tolist()




if __name__ == "__main__":
    example_title = "Toy Story (1995)"
    similar = get_similar_movies(example_title, k=10)
    print(f"\nMovies similar to '{example_title}':")
    for t in similar:
        print("-", t)

