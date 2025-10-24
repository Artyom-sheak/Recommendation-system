import pandas as pd
import os
df = pd.read_csv('/Users/artemcvetkov/Desktop/netflix_titles.csv')
df = df.drop('director', axis=1)
df['cast'] = df['cast'].fillna('Unknown cast')
df['country'] = df['country'].fillna('Unknown country')
df = df.dropna(subset = ['date_added'])
df = df.dropna(subset = ['rating','duration'])
df.info()
df['cast'] = df['cast'].apply( lambda x : ''.join(x.split(', ')[:3]) if x !='Unknown cast' else '')
df['content_features'] = (
    df['type'] + ' ' +
    df['listed_in'] + ' ' + df['listed_in'] +
    df['description'] + ' ' +
    df['cast'] + ' ' +
    df['country']
)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
    stop_words = 'english',
    max_features = 5000,
    ngram_range=(1,2)
)
tfidf_matrix = tfidf.fit_transform(df['content_features'])
print(f'Размерность TF-IDF матрицы: {tfidf_matrix.shape}')

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
print(f'Размерность матрицы схожести:{cosine_sim.shape}')

indices = pd.Series(df.index, index = df['title']).drop_duplicates()

def get_recommendations(title,cosine_sim=cosine_sim,top_n=10):

    if title not in indices:
        return f'Фильм "{title}" не найден в базе'

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices][['title','type','listed_in','description']]

    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return recommendations

print('===Тест рекомендаций===')
test_title = df['title'].iloc[0]
print(f'рекомендации для: {test_title}')
recommendations = get_recommendations(test_title)
print(recommendations[['title','similarity_score','listed_in']].head(5))

import pickle
os.makedirs('models',exist_ok=True)

model_artifacts = {
    'cosine_sim': cosine_sim,
    'indices': indices,
    'tfidf': tfidf,
    'df': df
}
with open('models/recomendations.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)


print('Модель сохранена!')
