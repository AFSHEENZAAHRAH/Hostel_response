import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai  # AI for structured reviews
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
import io
from fpdf import FPDF  # For PDF export
import xlsxwriter  # For Excel export
from datetime import datetime

# 🔐 **Basic Login System**
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login():
    st.title("🔑 Hostel Feedback Dashboard - Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":  # Simple auth system
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("🚨 Invalid credentials! Try again.")

if not st.session_state["logged_in"]:
    login()
    st.stop()

# 🏠 **Load and Preprocess Data**
file_path = r'C:\Users\afshe\OneDrive - Kumaraguru College of Technology\Desktop\AZ\PROJECTS\Hostel_response\Hostel_facility_response1.xlsx'
df = pd.read_excel(file_path, sheet_name="Form responses 1")

# 🔹 Rename Columns for Clarity
df.rename(columns={
    'Timestamp': 'Date',
    'Rate the overall quality of food served in the hostel mess': 'Food_Quality',
    'How would you rate the overall hostel facilities? ': 'Overall_Hostel_Rating',
    'How often are the washrooms cleaned in your hostel? ': 'Washroom_Cleaning',
    'Are the washrooms maintained properly  (no leakage, proper drainage, etc.)?': 'Washroom_Maintenance',
    'Any additional comments or suggestions for improving hostel facilities?': 'Comments'
}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

# 🔹 Convert Ratings to Numeric Scale
rating_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Very Good': 4, 'Excellent': 5}
df['Food_Quality'] = df['Food_Quality'].map(rating_map)
df['Overall_Hostel_Rating'] = df['Overall_Hostel_Rating'].map(rating_map)
df.dropna(subset=['Food_Quality', 'Overall_Hostel_Rating'], inplace=True)

# 🎯 **Advanced Data Filtering**
st.sidebar.subheader("🔍 Filter Data")
date_range = st.sidebar.date_input("📅 Select Date Range", [df["Date"].min(), df["Date"].max()])
selected_hostel_block = st.sidebar.selectbox("🏢 Select Hostel Block (if applicable)", ["All", "A Block", "B Block"])
complaint_filter = st.sidebar.text_input("❌ Search for Specific Complaints (e.g., WiFi, Food)")

# Apply Filters
df_filtered = df[(df["Date"] >= pd.to_datetime(date_range[0])) & (df["Date"] <= pd.to_datetime(date_range[1]))]
if selected_hostel_block != "All":
    df_filtered = df_filtered[df_filtered["Hostel_Block"] == selected_hostel_block]
if complaint_filter:
    df_filtered = df_filtered[df_filtered["Comments"].str.contains(complaint_filter, case=False, na=False)]

# 📊 **Data Visualizations**
st.title("🏠 Hostel Facility Feedback Dashboard")
plot_option = st.selectbox("📊 Select a Visualization:", [
    "Food Quality Rating (Bar Chart)",
    "Washroom Maintenance Feedback (Pie Chart)",
    "Overall Hostel Rating (Boxplot)",
    "Word Cloud for Comments",
    "Sentiment Analysis with Emojis",
    "Anomaly Detection in Ratings"
])

if plot_option == "Food Quality Rating (Bar Chart)":
    st.subheader("🍽️ Food Quality Rating Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(y=df['Food_Quality'], order=df['Food_Quality'].value_counts().index, hue=df['Food_Quality'], palette="coolwarm", legend=False, ax=ax)
    st.pyplot(fig)

if plot_option == "Word Cloud for Comments":
    st.subheader("📝 Common Words in Feedback")
    text = " ".join(df_filtered['Comments'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# 😊 **Sentiment Analysis with Emojis**
def get_sentiment_emoji(sentiment):
    if sentiment > 0.1:
        return "😃"
    elif sentiment < -0.1:
        return "😡"
    else:
        return "😐"

df_filtered["Sentiment"] = df_filtered["Comments"].dropna().astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
df_filtered["Sentiment_Emoji"] = df_filtered["Sentiment"].apply(get_sentiment_emoji)

if plot_option == "Sentiment Analysis with Emojis":
    st.subheader("📈 Sentiment Analysis with Emojis")
    st.dataframe(df_filtered[['Comments', 'Sentiment_Emoji']])

# 🤖 **AI-Generated Structured Hostel Review**
GOOGLE_API_KEY = "AIzaSyAoAo40jhkqcl542yr-7FZ6_Oo0Pf7p8II"
genai.configure(api_key=GOOGLE_API_KEY)

def generate_review(row):
    prompt = f"Write a structured hostel review for a student who rated Food: {row['Food_Quality']}/5, Hostel: {row['Overall_Hostel_Rating']}/5. Additional comments: {row['Comments']}"
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else "No review available."

df_filtered["AI_Review"] = df_filtered.apply(generate_review, axis=1)

if st.button("🤖 Generate Personalized AI Reviews"):
    st.subheader("📌 Personalized AI Reviews")
    st.dataframe(df_filtered[['Comments', 'AI_Review']])

# 📥 **Export Reports (PDF & Excel)**
if st.button("📤 Export as PDF & Excel"):
    df_filtered.to_excel("Hostel_Feedback.xlsx", index=False)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Hostel Feedback Report", ln=True, align='C')
    for i, row in df_filtered.iterrows():
        pdf.multi_cell(0, 10, f"{row['Comments']}: {row['AI_Review']}")
    pdf.output("Hostel_Feedback.pdf")

    st.success("✅ Reports Exported as PDF & Excel!")

