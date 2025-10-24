from fastapi import FastAPI
import pandas as pd
import pickle
app = FastAPI(title='Recommendation API')

with open('models/recomendations.pkl', 'rb') as f:
    model_data = pickle.load(f)

cosine_sim = model_data['cosine_sim']
indices = model_data['indices']
df = model_data['df']


def get_recommendations(title, cosine_sim=cosine_sim, top_n=10):
    if title not in indices:
        return {"error": f'Фильм "{title}" не найден в базе'}

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices][['title', 'type', 'listed_in', 'description']]
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return recommendations.to_dict('records')
@app.get('/')
async def root():
    return {'message': 'Recommendation API'}

@app.get('/recommend/{title}')
async def recommend(title: str,top_n:int = 10):
    recommendations = get_recommendations(title, cosine_sim, top_n)
    return {
        "requested_title": title,
        "recommendations": recommendations
    }

@app.get('/health')
async def health():
    return {'status': 'healthy'}


@app.get('/search/{query}')
async def search(query: str, limit: int = 5):
    """Поиск фильмов по названию"""
    matches = df[df['title'].str.contains(query, case=False, na=False)]
    results = matches[['title', 'type', 'listed_in']].head(limit)
    return results.to_dict('records')

