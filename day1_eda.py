import pandas as pd
from pathlib import Path

# 1. Point to your data folder
data_dir = Path("C:\\Users\\Srijal\\Downloads\\ml-latest-small")
# or Path("data/ml-latest-small")

ratings_path = data_dir / "ratings.csv"
movies_path = data_dir / "movies.csv"

# 2. Load CSVs
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

print("\nRatings head:")
print(ratings.head())

print("\nMovies head:")
print(movies.head())

# 3. Merge ratings with movies on movieId
ratings_movies = ratings.merge(movies, on="movieId", how="left")

print("\nMerged shape:", ratings_movies.shape)
print("\nMerged head:")
print(ratings_movies.head())

# 4. Save merged data to a fast format for later days
output_path = Path("data") / "ratings_movies.parquet"
output_path.parent.mkdir(exist_ok=True)
ratings_movies.to_parquet(output_path, index=False)

print("\nSaved merged file to:", output_path.resolve())
