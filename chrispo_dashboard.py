import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import random
import cv2
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
import os

# Download NLTK Stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("chrispo_data.csv")
        df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"])
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Ensure 'chrispo_data.csv' exists.")
        return pd.DataFrame()

df = load_data()

# Sidebar Navigation
st.sidebar.title("🏆 CHRISPO '25 Dashboard")
selected_section = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "📊 Participation Analysis", "📄 Text Analysis", "📷 Image Processing"]
)

# Function to load images safely
def load_image(image_name):
    img_path = os.path.join("images", image_name)
    return Image.open(img_path) if os.path.exists(img_path) else None

# 📌 HOME PAGE
if selected_section == "🏠 Home":
    st.title("🏠 Welcome to CHRISPO '25 Dashboard")

    st.markdown("""
    **CHRISPO '25** is the biggest inter-college sports tournament of the year!  
    This dashboard provides insights into participation, feedback analysis, and image processing of the event.  
    """)

    # 🚀 Quick Stats
    st.subheader("📌 Event Statistics")
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Participants", df.shape[0])
        with col2:
            st.metric("Unique Colleges", df["College"].nunique())
        with col3:
            top_sport = df["Sport"].mode()[0] if "Sport" in df.columns and not df.empty else "N/A"
            st.metric("Most Popular Sport", top_sport)
    else:
        st.warning("Dataset is empty. Check if 'chrispo_data.csv' is loaded correctly.")

    # 📸 Image Slideshow for Event Highlights
    st.subheader("📸 Event Highlights")
    event_images = ["sports1.jpeg", "sports2.jpeg"]
    
    img_index = st.slider("Scroll through images", 0, len(event_images) - 1, 0)
    image = load_image(event_images[img_index])

    if image:
        st.image(image, caption=f"Image {img_index+1} of {len(event_images)}", use_container_width=True)
    else:
        st.warning(f"Image {event_images[img_index]} not found!")

    st.markdown("---")
    st.subheader("📊 Explore More")
    st.markdown("Use the **sidebar navigation** to explore different sections of the dashboard!")

# 📊 PARTICIPATION ANALYSIS
elif selected_section == "📊 Participation Analysis":
    st.title("📊 CHRISPO '25 Participation Analysis")

    if df.empty:
        st.error("No data available for analysis.")
    else:
        # 🎯 Filters
        st.subheader("🎯 Filter Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_sport = st.selectbox("🏅 Select Sport", ["All"] + df["Sport"].unique().tolist())

        with col2:
            selected_college = st.selectbox("🏫 Select College", ["All"] + df["College"].unique().tolist())

        with col3:
            selected_state = st.selectbox("🌍 Select State", ["All"] + df["State"].unique().tolist())

        # 🔎 Apply Filters
        filtered_df = df.copy()
        if selected_sport != "All":
            filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
        if selected_college != "All":
            filtered_df = filtered_df[filtered_df["College"] == selected_college]
        if selected_state != "All":
            filtered_df = filtered_df[filtered_df["State"] == selected_state]

        st.subheader("📊 Filtered Data View")
        st.dataframe(filtered_df)

        # 📈 Charts Section
        st.subheader("📈 Data Visualization")
        col1, col2 = st.columns(2)

        with col1:
            # 🔥 Participation by Sport (Bar Chart)
            st.subheader("🏅 Sports Participation")
            sport_counts = filtered_df["Sport"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(sport_counts.index, sport_counts.values, color="skyblue")
            ax.set_xticks(range(len(sport_counts)))
            ax.set_xticklabels(sport_counts.index, rotation=45)
            ax.set_xlabel("Sport")
            ax.set_ylabel("Participants")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            st.pyplot(fig)

        with col2:
            # 📍 State-wise Participation (Pie Chart)
            st.subheader("🌎 State Participation")
            state_counts = filtered_df["State"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(state_counts, labels=state_counts.index, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
            st.pyplot(fig)

# 📄 TEXT ANALYSIS
elif selected_section == "📄 Text Analysis":
    st.title("📄 Text Analysis: Sports Feedback")

    # Handle missing feedback column
    if df.empty or "Feedback" not in df.columns or df["Feedback"].isna().all():
        st.error("No feedback data available.")
    else:
        # 🎯 Sport Selection
        st.sidebar.subheader("🎯 Select a Sport")
        selected_sport = st.sidebar.selectbox("🏅 Choose Sport", df["Sport"].unique().tolist())

        # Filter feedback for the selected sport
        sport_df = df[df["Sport"] == selected_sport]
        sport_feedback = " ".join(sport_df["Feedback"].dropna())

        # 📌 Generate Word Clouds
        st.subheader(f"🎨 Word Clouds for {selected_sport} & Other Sports")

        col1, col2, col3 = st.columns(3)

        def generate_wordcloud(text, colormap):
            if text:
                words = [word for word in text.lower().split() if word not in stop_words]
                cleaned_text = " ".join(words)
                return WordCloud(width=500, height=300, background_color="white", colormap=colormap).generate(cleaned_text)
            else:
                return None

        with col1:
            st.markdown(f"**{selected_sport}**")
            wordcloud = generate_wordcloud(sport_feedback, "coolwarm")
            if wordcloud:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning(f"No feedback available for {selected_sport}.")

        # Generate Word Clouds for top 2 most reviewed sports (excluding the selected one)
        top_sports = df["Sport"].value_counts().index.tolist()
        top_sports = [sport for sport in top_sports if sport != selected_sport][:2]

        for i, sport in enumerate(top_sports):
            sport_df = df[df["Sport"] == sport]
            sport_feedback = " ".join(sport_df["Feedback"].dropna())

            col = col2 if i == 0 else col3
            with col:
                st.markdown(f"**{sport}**")
                wordcloud = generate_wordcloud(sport_feedback, "viridis" if i == 0 else "plasma")
                if wordcloud:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning(f"No feedback available for {sport}.")

        # 📊 Feedback Comparison Table
        st.subheader(f"💬 Feedback Samples for {selected_sport}")

        feedback_list = sport_df["Feedback"].dropna().tolist()

        if feedback_list:
            random_feedback = random.sample(feedback_list, min(5, len(feedback_list)))
            for i, feedback in enumerate(random_feedback):
                st.markdown(f"**{i+1}.** {feedback}")
        else:
            st.warning(f"No feedback available for {selected_sport}.")

        # 📊 Sport-Wise Feedback Summary
        st.subheader("📊 Sports-wise Feedback Comparison")

        sport_feedback_counts = df.groupby("Sport")["Feedback"].count().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(sport_feedback_counts.index, sport_feedback_counts.values, color="skyblue")
        ax.set_xlabel("Number of Feedback Entries")
        ax.set_ylabel("Sport")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        st.pyplot(fig)

#imageprocessing
elif selected_section == "📷 Image Processing":
    st.title("📷 Image Processing")

    # 📅 Day-wise Image Gallery
    st.subheader("📅 Day-wise Image Gallery")

    # Load images from the "images" folder
    image_folder = "images"
    img1 = Image.open(os.path.join(image_folder, "sports1.jpeg"))
    img2 = Image.open(os.path.join(image_folder, "sports2.jpeg"))
    
    gallery_images = [img1, img2] * 3  # Repeat images for gallery

    cols = st.columns(3)  # Display images in 3 columns
    for i, img in enumerate(gallery_images):
        with cols[i % 3]:
            st.image(img, caption=f"Day {i+1}", use_container_width=True)

    # 🎨 Custom Image Processing
    st.subheader("🎨 Custom Image Processing")
    
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        # 📌 Multiple Filter Selection
        filters = st.multiselect("Choose Filters", ["Grayscale", "Blur", "Edge Detection"])

        # Convert image to numpy array
        img_array = np.array(image)

        if "Grayscale" in filters:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        if "Blur" in filters:
            blur_intensity = st.slider("Blur Intensity", 1, 25, 5, step=2)
            img_array = cv2.GaussianBlur(img_array, (blur_intensity, blur_intensity), 0)

        if "Edge Detection" in filters:
            edge_low = st.slider("Edge Detection - Low Threshold", 50, 150, 100)
            edge_high = st.slider("Edge Detection - High Threshold", 100, 300, 200)
            img_array = cv2.Canny(img_array, edge_low, edge_high)

        st.image(img_array, caption="Processed Image", use_container_width=True, 
                 channels="GRAY" if "Grayscale" in filters else "RGB")
