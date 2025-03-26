# type: ignore
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load dataset
@st.cache_data
def load_data():
    restaurants = pd.read_csv("restaurants1.csv")
    interactions = pd.read_csv("interactions.csv")
    return restaurants, interactions

restaurants, interactions = load_data()

# Popularity-Based Data Preparation
pdata = restaurants[['name', 'rating', 'price_range']]

# Content-Based Data Preparation
cdata = restaurants[['name', 'cuisine', 'location', 'rating', 'price_range']].copy()

# Categorical Encoding
for col in ['name', 'cuisine', 'location']:
    order = cdata.groupby(col)['rating'].sum().sort_values().index
    orders = {val: indx + 1 for indx, val in enumerate(order)}
    cdata[col] = cdata[col].replace(orders)

# Train NearestNeighbors model
cosinemodel = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 includes input item
cosinemodel.fit(cdata.drop(columns=['rating', 'price_range']))

# Function to recommend similar restaurants
def recommend_similar_restaurants(restaurantname, n=5):
    restaurantname = restaurantname.lower().strip()
    idx = restaurants[restaurants['name'].str.lower().str.strip() == restaurantname].index.tolist()
    
    if not idx:
        return []
    
    idx = idx[0]
    data = cdata.drop(columns=['rating', 'price_range']).iloc[idx].values.reshape(1, -1)
    distances, indices = cosinemodel.kneighbors(data, n_neighbors=n+1)
    return indices.flatten()[1:].tolist()  # Skip input item

# Streamlit UI
st.subheader("🍽️ Restaurants Recommendations")
st.subheader("🔥 Popular Restaurants")

# Taking Top 10 based on Popularity Score
indxs = pdata.sort_values(by='rating', ascending=False).head(10).index.tolist()
name_list = restaurants.loc[indxs, 'name'].str.title().tolist()

selected_restaurant = st.selectbox("Select a restaurant:", name_list)

if selected_restaurant:
    st.session_state["selected_section"] = selected_restaurant

st.divider()

if "selected_section" in st.session_state:
    selected_restaurant = st.session_state["selected_section"]
    selected_data = restaurants[restaurants['name'].str.title() == selected_restaurant].iloc[0]

    st.subheader("✅ Selected Restaurant")
    st.write(f"🍽️ **Restaurant Name:** {selected_data['name']}")
    st.write(f"📍 **Location:** {selected_data['location']}")
    st.write(f"⭐ **Rating:** {selected_data['rating']}")
    st.write(f"💰 **Price Range:** ${selected_data['price_range']}")

    if st.button("🔍 Show Similar Restaurants"):
        st.divider()
        st.subheader("🍽️ Similar Restaurants")
        
        restaurant_name = selected_data['name'].lower()
        indxs = recommend_similar_restaurants(restaurant_name)
        
        if not indxs:
            st.error("❌ No similar restaurants found.")
        else:
            recommended_names = restaurants.loc[indxs, 'name'].str.title().tolist()
            for name in recommended_names:
                st.write(f"🍽️ {name}")
