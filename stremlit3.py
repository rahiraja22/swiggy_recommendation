import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the final dataset (combined cleaned + encoded data)
df = pd.read_csv('final_10000_combined_encoded_data.csv')

# Extract original columns for display and filtering
original_columns = ['name', 'cuisine', 'main_city', 'area', 'rating', 'cost']
encoded_columns = [col for col in df.columns if col not in original_columns]

# Compute similarity from encoded part
similarity = cosine_similarity(df[encoded_columns])

# Streamlit UI
st.title("🍽️ Swiggy Restaurant Recommendation System")

# User Input
city = st.selectbox("Select City", sorted(df['main_city'].dropna().unique()))
area = st.selectbox("Select Area", sorted(df[df['main_city'] == city]['area'].dropna().unique()))
cuisine = st.selectbox("Preferred Cuisine", sorted(df['cuisine'].dropna().unique()))
rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, step=0.1)
max_cost = st.slider("Maximum Cost", 100, 1000, 300, step=50)

# Filter based on user input
filtered_df = df[
    (df['main_city'] == city) &
    (df['area'] == area) &
    (df['cuisine'] == cuisine) &
    (df['rating'].apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0.0) >= rating) &
    (df['cost'] <= max_cost)
]

def recommend(index):
    if index >= len(similarity):
        return pd.DataFrame()
    scores = similarity[index]
    top_indices = scores.argsort()[::-1][1:6]
    return df.iloc[top_indices][original_columns]

# Recommendation Button
if st.button("Recommend Restaurants"):
    if not filtered_df.empty:
        selected_index = filtered_df.index[0]
        st.success("🍴 Here are your recommendations:")
        st.dataframe(recommend(selected_index))
    else:
        st.warning("😔 No restaurants found matching your criteria.")



