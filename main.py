# Imports
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def format_title(title: str) -> str:
    return title.strip().title()
    
    
def get_movie_index(movie_name: str) -> int:
    return movies_pivot.reset_index()[
        movies_pivot.reset_index()['title'] == movie_name
        ].index


def get_suggestions_movies_index(movie_name: str = None, movie_index: int = None) -> list[int] or None:
    if movie_name:
        movie_index = get_movie_index(movie_name=movie_name)
        distances, suggestions_index = model.kneighbors(
            movies_pivot.iloc[movie_index, :].values.reshape(1, -1)
        )

        return suggestions_index
    elif movie_index:
        distances, suggestions_index = model.kneighbors(
            movies_pivot.iloc[movie_index, :].values.reshape(1, -1)
        )

        return suggestions_index
    else:
        return None


def predict(movie_name: str = None, movie_index: int = None) -> list[str]:
    suggestions_names = list()

    if movie_name:
        movie_name = movie_name.strip().title()

    suggestions_index = get_suggestions_movies_index(
        movie_name=movie_name,
        movie_index=movie_index
    )

    for movie_index in suggestions_index:
        suggestions_names.append(movies_pivot.index[movie_index])

    suggestions_names = list(suggestions_names[0])
    suggestions_names.pop(0)

    return suggestions_names


# Get Data
user_df = pd.read_csv('data/Dataset.csv')
movie_title_df = pd.read_csv('data/Movie_Id_Titles.csv')

# Standardizing the 'titles'
movie_title_df['title'] = movie_title_df['title'].apply(format_title)

# Merge
movies_df = pd.merge(user_df, movie_title_df, on='item_id')

# Users with more 100 reviews
user_above_100_ratings = movies_df['user_id'].value_counts() > 100
user_above_100_ratings = user_above_100_ratings[user_above_100_ratings].index
movies_df = movies_df[movies_df['user_id'].isin(user_above_100_ratings)]

# Number of reviews for each movie
number_of_ratings = movies_df.groupby('title')['rating'].count().reset_index()
number_of_ratings.rename(columns={'rating': 'number_of_ratings'}, inplace=True)

# Merge with Number of ratings
movies_df = movies_df.merge(number_of_ratings, on='title')

# Movies with more 50 ratings
movies_df = movies_df[movies_df['number_of_ratings'] >= 50]

# Delete(drop) duplicates values
movies_df.drop_duplicates(subset=['user_id', 'title'], inplace=True)

# Transposition of rows(user_id) into columns
movies_pivot = movies_df.pivot_table(
    columns='user_id',
    index='title',
    values='rating'
)

# Fill the NaN values
movies_pivot.fillna(0, inplace=True)

# Converting to a sparce matrix
movie_sparce = csr_matrix(movies_pivot)

# Create and training the model
model = NearestNeighbors(algorithm='brute')
model.fit(movie_sparce)

if __name__ == '__main__':
    movie_name = 'Adventures Of Priscilla, Queen Of The Desert, The (1994)'
    
    suggestions = predict(movie_name=movie_name)
    
    print(f'\nMovie: {movie_name}\n\nSuggestions:')
    for movie in suggestions:
        print(f'→ {movie}')
    print('\n')
