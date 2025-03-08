import os
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import pandas as pd
import io
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
UNSPLASH_API_KEY = st.secrets.get("unsplash_api_key")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini Flash model: {e}")
    st.stop()

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["source", "destination"],
    template="""
    You are a travel planning assistant. Provide travel options from {source} to {destination}.
    Present the information in a structured table format with the following columns:

    | Travel Type | Price (Estimated) | Time (Estimated) | Description | Comfort Level (1-5, 5 being highest) | Directness (Direct/Indirect) |
    |-------------------|-------------------|-------------------|-------------|------------------------------------|-----------------------------|
    | Cab/Taxi          |                   |                   |             |                                    |                             |
    | Train             |                   |                   |             |                                    |                             |
    | Bus               |                   |                   |             |                                    |                             |
    | Flight            |                   |                   |             |                                    |                             |
    | Ola/Uber          |                   |                   |             |                                    |                             |

    Fill in the table with estimated prices, travel times, descriptions, comfort levels (1-5), and directness.
    If a mode of transport is unavailable, indicate it in the table.
    """
)

travel_chain = LLMChain(llm=llm, prompt=prompt_template)

def get_travel_recommendations(source, destination):
    try:
        response = travel_chain.run({"source": source, "destination": destination})
        return response if isinstance(response, str) else response["text"]
    except Exception as e:
        return f"An error occurred: {e}"

# Fetch images from Unsplash
def fetch_unsplash_images(query, count=3):
    url = f"https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "per_page": count,
        "client_id": UNSPLASH_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return [img['urls']['regular'] for img in data['results']]
    return []

# UI
st.title("🌍 AI-Powered Travel Planner")
st.write("Plan your trip with AI-powered travel options and beautiful images.")

# Input fields
source = st.text_input("🏙️ Enter Source City:")
destination = st.text_input("📍 Enter Destination City:")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date")
with col2:
    end_date = st.date_input("📅 End Date")

preferred_transport = st.selectbox("✈️ 🚂 🚌 Preferred Transport Mode", ["Flight", "Train", "Bus", "Cab/Taxi", "Ola/Uber"])
budget_range = st.selectbox("💰 Budget Range", ["Budget", "Mid-range", "Luxury"])
preferred_time = st.selectbox("⏰ Preferred Time to Travel", ["Morning", "Afternoon", "Evening", "Night"])
num_travelers = st.number_input("👥 Number of Travelers", min_value=1, value=1)
preferred_currency = st.selectbox("💱 Preferred Currency", ["INR", "USD", "EUR"])

if st.button("✨ Get Travel Options"):
    if source and destination:
        st.write(f"Generating travel options from **{source}** to **{destination}**...")
        
        # Fetch 3 images from Unsplash
        images = fetch_unsplash_images(destination, count=3)
        if images:
            st.subheader(f"📸 Stunning Views of {destination}")
            cols = st.columns(3)
            for i, img_url in enumerate(images):
                response = requests.get(img_url)
                image = Image.open(BytesIO(response.content))
                with cols[i]:
                    st.image(image, caption=f"View {i+1}", use_container_width=True)

        # Get AI travel recommendations
        recommendations = get_travel_recommendations(source, destination)
        st.write("### 🗺️ Travel Recommendations")
        st.write(recommendations)

        # Process data into DataFrame for visualization
        try:
            table_data = recommendations.strip().split('\n')[2:-1]
            rows = [row.strip().split('|')[1:-1] for row in table_data]
            df = pd.DataFrame(rows, columns=["Travel Type", "Price (Estimated)", "Time (Estimated)", "Description", "Comfort Level", "Directness"])

            df["Price (Estimated)"] = pd.to_numeric(df["Price (Estimated)"].str.replace(r'[^\d\.]+', '', regex=True), errors='coerce')
            df["Time (Estimated)"] = pd.to_numeric(df["Time (Estimated)"].str.replace(r'[^\d\.]+', '', regex=True), errors='coerce')

            # Price Chart
            fig_price = go.Figure([go.Bar(x=df["Travel Type"], y=df["Price (Estimated)"])])
            fig_price.update_layout(title="💰 Price Comparison", xaxis_title="Travel Type", yaxis_title="Price")
            st.plotly_chart(fig_price)

            # Time Chart
            fig_time = go.Figure([go.Bar(x=df["Travel Type"], y=df["Time (Estimated)"])])
            fig_time.update_layout(title="⏳ Time Comparison", xaxis_title="Travel Type", yaxis_title="Time (hrs)")
            st.plotly_chart(fig_time)

            # Combined Chart
            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(x=df["Travel Type"], y=df["Price (Estimated)"], name="Price", mode='lines+markers'))
            fig_combined.add_trace(go.Scatter(x=df["Travel Type"], y=df["Time (Estimated)"], name="Time", mode='lines+markers'))
            fig_combined.update_layout(title="💵 Price & Time Comparison", xaxis_title="Travel Type", yaxis_title="Value")
            st.plotly_chart(fig_combined)

            # CSV Download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(label="📥 Download Travel Data as CSV", data=csv_buffer.getvalue(), file_name="travel_data.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing data: {e}")
    else:
        st.error("Please enter both source and destination cities.")

# Sidebar Info
st.sidebar.header("ℹ️ Project Info")
st.sidebar.write("""
This app uses Google Gemini AI & Unsplash API to suggest travel options and show beautiful images of your destination.
""")

st.sidebar.subheader("⚙️ Technologies")
st.sidebar.write("""
- Python
- Streamlit
- LangChain (Google Gemini Flash)
- Unsplash API
- pandas & plotly
""")
