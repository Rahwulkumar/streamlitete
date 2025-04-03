# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import nltk
# import random
# import cv2
# from nltk.corpus import stopwords
# from wordcloud import WordCloud
# from PIL import Image
# import os

# # Download NLTK Stopwords
# nltk.download("stopwords")
# stop_words = set(stopwords.words("english"))

# # Load dataset
# @st.cache_data
# def load_data():
#     return pd.read_csv("chrispo_data.csv")

# df = load_data()
# df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"])

# # Sidebar Navigation
# st.sidebar.title("🏆 CHRISPO '25 Dashboard")
# selected_section = st.sidebar.radio(
#     "Go to", 
#     ["🏠 Home", "📊 Participation Analysis", "📄 Text Analysis", "📷 Image Processing"]
# )

# # Function to load images from the correct path
# def load_image(image_name):
#     img_path = os.path.join("images", image_name)
#     return Image.open(img_path) if os.path.exists(img_path) else None

# # 📌 HOME PAGE
# if selected_section == "🏠 Home":
#     st.title("🏠 Welcome to CHRISPO '25 Dashboard")

#     st.markdown("""
#     **CHRISPO '25** is the biggest inter-college sports tournament of the year!  
#     This dashboard provides insights into participation, feedback analysis, and image processing of the event.  
#     """)

#     # 🚀 Quick Stats
#     st.subheader("📌 Event Statistics")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Participants", df.shape[0])
#     with col2:
#         st.metric("Unique Colleges", df["College"].nunique())
#     with col3:
#         top_sport = df["Sport"].mode()[0] if not df.empty else "N/A"
#         st.metric("Most Popular Sport", top_sport)

#     # 📸 Image Slideshow for Event Highlights
#     st.subheader("📸 Event Highlights")
#     event_images = ["sports1.jpeg", "sports2.jpeg"]
    
#     img_index = st.slider("Scroll through images", 0, len(event_images) - 1, 0)
#     image = load_image(event_images[img_index])

#     if image:
#         st.image(image, caption=f"Image {img_index+1} of {len(event_images)}", use_container_width=True)
#     else:
#         st.warning(f"Image {event_images[img_index]} not found!")

#     st.markdown("---")
#     st.subheader("📊 Explore More")
#     st.markdown("Use the **sidebar navigation** to explore different sections of the dashboard!")

# # 📊 PARTICIPATION ANALYSIS
# elif selected_section == "📊 Participation Analysis":
#     st.title("📊 CHRISPO '25 Participation Analysis")

#     # 🎯 Filters
#     st.subheader("🎯 Filter Data")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         selected_sport = st.selectbox("🏅 Select Sport", ["All"] + df["Sport"].unique().tolist())

#     with col2:
#         selected_college = st.selectbox("🏫 Select College", ["All"] + df["College"].unique().tolist())

#     with col3:
#         selected_state = st.selectbox("🌍 Select State", ["All"] + df["State"].unique().tolist())

#     # 🔎 Apply Filters
#     filtered_df = df.copy()
#     if selected_sport != "All":
#         filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
#     if selected_college != "All":
#         filtered_df = filtered_df[filtered_df["College"] == selected_college]
#     if selected_state != "All":
#         filtered_df = filtered_df[filtered_df["State"] == selected_state]

#     st.subheader("📊 Filtered Data View")
#     st.dataframe(filtered_df)

#     # 📈 Charts Section
#     st.subheader("📈 Data Visualization")
#     col1, col2 = st.columns(2)

#     with col1:
#         # 🔥 Participation by Sport (Bar Chart)
#         st.subheader("🏅 Sports Participation")
#         sport_counts = filtered_df["Sport"].value_counts()
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.bar(sport_counts.index, sport_counts.values, color="skyblue")
#         ax.set_xlabel("Sport")
#         ax.set_ylabel("Participants")
#         ax.set_xticklabels(sport_counts.index, rotation=45)
#         ax.grid(axis="y", linestyle="--", alpha=0.7)
#         st.pyplot(fig)

#     with col2:
#         # 📍 State-wise Participation (Pie Chart)
#         st.subheader("🌎 State Participation")
#         state_counts = filtered_df["State"].value_counts()
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.pie(state_counts, labels=state_counts.index, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
#         st.pyplot(fig)

#     col3, col4 = st.columns(2)

#     with col3:
#         # 🏫 Top 5 Colleges with Highest Participation
#         st.subheader("🏫 Top 5 Colleges")
#         top_colleges = filtered_df["College"].value_counts().nlargest(5)
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.barh(top_colleges.index, top_colleges.values, color="orange")
#         ax.set_xlabel("Number of Participants")
#         ax.set_ylabel("College")
#         ax.invert_yaxis()
#         st.pyplot(fig)

#     with col4:
#         # 📆 Participation Over Time (Line Chart)
#         st.subheader("📅 Participation Trend")
#         date_counts = filtered_df.groupby(filtered_df["RegistrationDate"].dt.date).size()
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.plot(date_counts.index, date_counts.values, marker="o", linestyle="-", color="green")
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Participants")
#         ax.grid(True, linestyle="--", alpha=0.7)
#         st.pyplot(fig)

#     # 📊 Sport-wise Gender Distribution (Stacked Bar Chart)
#     st.subheader("⚡ Sport-wise Gender Distribution")
#     gender_counts = filtered_df.groupby(["Sport", "Gender"]).size().unstack()
    
#     if not gender_counts.empty:
#         fig, ax = plt.subplots(figsize=(7, 4))
#         gender_counts.plot(kind="bar", stacked=True, ax=ax, color=["blue", "pink"])
#         ax.set_xlabel("Sport")
#         ax.set_ylabel("Count")
#         ax.legend(title="Gender")
#         ax.grid(axis="y", linestyle="--", alpha=0.7)
#         st.pyplot(fig)
#     else:
#         st.warning("No data available for the selected filters!")


# # 📄 TEXT ANALYSIS
# elif selected_section == "📄 Text Analysis":
#     st.title("📄 Text Analysis: Sports Feedback")

#     st.subheader("📝 Word Clouds")
#     sport_feedback = " ".join(df["Feedback"].dropna())

#     if sport_feedback:
#         words = [word for word in sport_feedback.lower().split() if word not in stop_words]
#         cleaned_text = " ".join(words)
#         wordcloud = WordCloud(width=500, height=300, background_color="white").generate(cleaned_text)

#         fig, ax = plt.subplots()
#         ax.imshow(wordcloud, interpolation="bilinear")
#         ax.axis("off")
#         st.pyplot(fig)
#     else:
#         st.warning("No feedback available.")

#     # Feedback Comparison
#     st.subheader("💬 Feedback Samples")
#     feedback_list = df["Feedback"].dropna().tolist()

#     if feedback_list:
#         st.markdown("**Random Feedback Samples:**")
#         random_feedback = random.sample(feedback_list, min(3, len(feedback_list)))
#         for feedback in random_feedback:
#             st.text_area(label="", value=feedback, height=80, key=feedback[:10])
#     else:
#         st.warning("No feedback available.")

# # 📷 IMAGE PROCESSING
# elif selected_section == "📷 Image Processing":
#     st.title("📷 Image Processing")

#     # 📷 Image Gallery
#     st.subheader("📅 Event Images")
#     cols = st.columns(2)
#     with cols[0]:
#         img1 = load_image("sports1.jpeg")
#         if img1:
#             st.image(img1, caption="Sports 1", width=250)
#         else:
#             st.warning("sports1.jpeg not found!")
    
#     with cols[1]:
#         img2 = load_image("sports2.jpeg")
#         if img2:
#             st.image(img2, caption="Sports 2", width=250)
#         else:
#             st.warning("sports2.jpeg not found!")

#     # 🖼️ Custom Image Processing
#     st.subheader("🖌️ Apply Filters")
#     uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Original Image", use_container_width=True)

#         option = st.selectbox("Choose a Filter", ["Grayscale", "Blur", "Edge Detection"])

#         if option == "Grayscale":
#             processed_image = image.convert("L")
#         elif option == "Blur":
#             processed_image = cv2.GaussianBlur(np.array(image), (15, 15), 0)
#             processed_image = Image.fromarray(processed_image)
#         elif option == "Edge Detection":
#             processed_image = Image.fromarray(cv2.Canny(np.array(image), 100, 200))

#         st.image(processed_image, caption="Processed Image", use_container_width=True)
