# type: ignore
import streamlit as st
from st_clickable_images import clickable_images # pip install st_clickable_images
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse 



# Load MovieLens dataset (Ensure this dataset has an 'image_url' column)
@st.cache_data
def load_data():
    items= pd.read_csv("cellphones_dataurl1.csv")
    interactions= pd.read_csv("cellphones ratings.csv")
    users= pd.read_csv("cellphones users.csv")
    return items, interactions,users

items, interactions,users = load_data()

################################################# Popularity Recommendations Pre-Work ###########################################
pdata = pd.merge(items,interactions, on='cellphone_id')

################################################# Content Recommendations Pre-Work ###########################################
# Load the data (Use `items` DataFrame)
df = items.copy()

# Combine relevant text features into a single string for each phone
df["features"] = df["brand"] + " " + df["model"] + " " + df["operating system"] + " " + \
                 df["internal memory"].astype(str) + "GB storage " + df["RAM"].astype(str) + "GB RAM " + \
                 df["main camera"].astype(str) + "MP main camera " + df["selfie camera"].astype(str) + "MP selfie camera " + \
                 df["battery size"].astype(str) + "mAh battery " + df["screen size"].astype(str) + "-inch screen " + \
                 df["weight"].astype(str) + "g weight " + "$" + df["price"].astype(str) + " price"

# Convert text features into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
text_features = vectorizer.fit_transform(df["features"])

# Normalize numerical features (price, RAM, battery size)
scaler = StandardScaler()
numerical_features = scaler.fit_transform(df[["RAM", "battery size", "weight", "price", "internal memory", "performance", "screen size"]])

# Combine text and numerical features
combined_features = scipy.sparse.hstack((text_features, numerical_features)) #scipy.sparse.hstack() horizontally stacks (joins) text features and numerical features into one large feature matrix.
# Convert to indexable sparse matrix
combined_features = combined_features.tocsr()  # Convert to CSR format csr_matrix (Compressed Sparse Row Format),Efficient for row-based indexing (which we need).
knn = NearestNeighbors(n_neighbors=6, metric="cosine")  # K=6 (1 for itself + 5 recommendations)
knn.fit(combined_features)
# Function to recommend movies
def recommend_knn_cosine(model_name, df, knn_model, feature_matrix, n=5):    
    # Convert model name to lowercase for consistency
    model_name = model_name
    # Check if the model exists in the dataset
    model_index = df[df["model"] == model_name].index
    if len(model_index) == 0:
        return f"Model '{model_name}' not found in the dataset."
    
    model_index = model_index[0]   # Take first match if multiple exist
    
    # Find N nearest neighbors
    distances, indices = knn_model.kneighbors(feature_matrix[model_index].reshape(1, -1), n_neighbors=n+1)
    recommended_indices = indices.flatten()[1:]   # Skip input 
    
    return recommended_indices

################################################################# Streamlit UI #######################################################
st.subheader(":blue[🏷️ 🏷️ 🏷️ Cell Phones Recommendations 🏷️ 🏷️ 🏷️]", divider=True)
st.subheader(":red[Popular cell phones..!]")

# Taking Top 10 based on Popularity Score
indxs = pdata.sort_values(by="rating", ascending=False)[0:10].index


# Display images as clickable grid
image_paths = [pdata['url'].iloc[i] for i in indxs]
brand= [pdata['brand'].iloc[i] for i in indxs]
model = [pdata['model'].iloc[i] for i in indxs]
prices = [pdata['price'].iloc[i] for i in indxs]
Internal_memory = [pdata['internal memory'].iloc[i] for i in indxs]
RAM = [pdata['RAM'].iloc[i] for i in indxs]
main_camera = [pdata['main camera'].iloc[i] for i in indxs]
names = [f"{b} {m}" for b, m in zip(brand, model)]

selected_index = clickable_images(
    image_paths,
    titles=names,
    div_style={"display": "flex", "flex-wrap": "wrap", "gap": "30px"},
    img_style={"height": "90px", "border-radius": "10px", "cursor": "pointer"}
)

# Store selection
if selected_index is not None:
    st.session_state["selected_section"] = selected_index + 1

st.divider()

if "selected_section" in st.session_state:
    st.subheader(":green[Selected..]")
    indx = st.session_state['selected_section']-1
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image_paths[indx], caption=model[indx])
    st.write(":green[Brand:]",brand[indx])

    st.write(":green[Model:]", model[indx])
    st.write(":green[Price: $ ]", prices[indx])
    st.write(":green[RAM:]", RAM[indx])
    st.write(":green[Main camera: ]",main_camera[indx])
    st.write(":green[Internal memory: ]",Internal_memory[indx])


    if st.button("Recommend Similar Items:"):
        st.divider()
        st.subheader(":red[Similar cell phones..!]")
        title = model[indx]
        indxs = recommend_knn_cosine(title,df, knn, combined_features, n=5)
        # Display images as clickable grid
        image_paths = [items['url'].iloc[i] for i in indxs]
        brand= [items['brand'].iloc[i] for i in indxs]
        model = [items['model'].iloc[i] for i in indxs]
        prices = [items['price'].iloc[i] for i in indxs]
        Internal_memory = [items['internal memory'].iloc[i] for i in indxs]
        RAM = [items['RAM'].iloc[i] for i in indxs]
        main_camera = [items['main camera'].iloc[i] for i in indxs]
        names = [f"{b} {m}" for b, m in zip(brand, model)]


        selected_index2 = clickable_images(image_paths,titles=names,
                                          div_style={"display": "flex", "flex-wrap": "wrap", "gap": "30px"},
                                          img_style={"height": "200px", "border-radius": "10px", "cursor": "pointer"}
                                          )
