import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================= LOAD MODELS & DATA =================
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("product_similarity.pkl", "rb") as f:
    product_similarity = pickle.load(f)

rfm_clusters = pd.read_csv("rfm_with_clusters.csv")

# ================= HELPER FUNCTIONS =================
def recommend_products(product_name, n=5):
    if product_name not in product_similarity.columns:
        return None
    similar_products = product_similarity[product_name].sort_values(ascending=False).drop(product_name).head(n)
    return similar_products.index.tolist()

def predict_cluster(recency, frequency, monetary):
    X = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(X)[0]
    # Map cluster to segment label from saved RFM
    segment = rfm_clusters.loc[rfm_clusters['Cluster'] == cluster, 'Segment'].iloc[0]
    return segment

# ================= STREAMLIT APP =================
st.title("🛒 Shopper Spectrum App")
st.write("Customer Segmentation & Product Recommendation")

# Tabs
tab1, tab2 = st.tabs(["📦 Product Recommendation", "👥 Customer Segmentation"])

# ========== Product Recommendation Module ==========
# ========== Product Recommendation Module ==========
# ========== Product Recommendation Module ==========
with tab1:
    st.header("Product Recommendation")

    # Load all valid products
    product_names = sorted(product_similarity.columns.tolist())

    # First dropdown: pick a base product
    product_name = st.selectbox("Select a Product:", options=product_names)

    if st.button("Get Recommendations"):
        recommendations = recommend_products(product_name)
        if recommendations:
            st.success("Top 5 similar products:")
            for i, prod in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {prod}**")

            # Second dropdown: pick one of the recommended products
            recommended_product = st.selectbox(
                "Select from recommended products:",
                options=recommendations
            )
            st.info(f"You selected recommended product: **{recommended_product}**")
        else:
            st.error(f"Product '{product_name}' not found in data.")

# ========== Customer Segmentation Module ==========
with tab2:
    st.header("Customer Segmentation")
    recency = st.number_input("Recency (days)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=10)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Cluster"):
        segment = predict_cluster(recency, frequency, monetary)
        st.success(f"Predicted Customer Segment: **{segment}**")
