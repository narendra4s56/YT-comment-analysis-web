import streamlit as st
import googleapiclient.discovery
import googleapiclient.errors
from urllib.parse import urlparse, parse_qs
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

# Load models
with open('spam_model.pkl', 'rb') as file1:
    spam_model = pickle.load(file1)

with open('sentiment_classifier_model.pkl', 'rb') as file2:
    sentiment_model = pickle.load(file2)
with open('sentiment_vectorizer.pkl', 'rb') as f:
    sentiment_vectorizer = pickle.load(f)

with open('toxic_vectorizer.pkl', 'rb') as file3:
    vectorizer = pickle.load(file3)
toxic_model = load_model('toxic_comment_model.h5')


# Helper function: Time series plot of comment counts over time per class
def plot_time_series(df):
    if df.empty or 'DATE' not in df.columns or 'PREDICTION' not in df.columns:
        st.write("No data for time series plot.")
        return
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    counts = df.groupby(['DATE', 'PREDICTION']).size().unstack(fill_value=0)
    st.line_chart(counts)


# Helper function: Word frequency heatmap for comments by prediction label



# --- Streamlit UI ---
st.set_page_config(page_title='YouTube Comment Analyzer', layout="wide")
st.title(':rainbow[YouTube Comment Analysis]')

# --- Input Section ---
url = st.text_input("🎥 Enter or paste a YouTube video URL:")

# --- Setup Tabs ---
tab1, tab2, tab3 = st.tabs(["📨 Spam Analyzer", "😊 Sentiment Analyzer", "☢️ Toxicity Analyzer"])

# --- Check for valid YouTube URL ---
video_id = None
if url:
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "v" in query_params:
        video_id = query_params["v"][0]
    else:
        st.error("❌ Invalid YouTube URL. Please make sure it includes '?v=VIDEO_ID'.")


# --- Function to Fetch Comments ---
def get_comments(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyD46zmOcOwy_lrHWuVCy7m4L2loaeSdnLY"  # Replace with your key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY
    )

    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()
    video_title = video_response['items'][0]['snippet']['title']

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()

    comments_data = []
    texts = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        content = comment.get('textDisplay')
        author = comment.get('authorDisplayName')
        date = comment.get('publishedAt')

        comments_data.append({
            "AUTHOR": author,
            "DATE": date,
            "CONTENT": content,
            "VIDEO_NAME": video_title
        })
        texts.append(content)

    return pd.DataFrame(comments_data), texts


# --- TAB 1: Spam Analyzer ---
with tab1:
    st.subheader("📨 Spam Analyzer")

    if url and video_id:
        df, texts = get_comments(video_id)
        predictions = spam_model.predict(texts)
        prediction_labels = ['Spam' if p == 1 else 'Not Spam' for p in predictions]
        df['PREDICTION'] = prediction_labels

        st.dataframe(df)

        count_data = df['PREDICTION'].value_counts()

        # Row 1: Pie chart | Bar chart
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(
                count_data,
                labels=count_data.index,
                startangle=90,
                shadow=True,
                autopct='%.2f%%',
                colors=['#800000', '#508D4E'],
                wedgeprops=dict(alpha=0.8, edgecolor='#6C48C5'),
                textprops={'fontsize': 12}
            )
            centre_circle = plt.Circle((0, 0), 0.10, fc='white', linewidth=1.25, edgecolor='black')
            fig.gca().add_artist(centre_circle)
            plt.axis('equal')
            plt.title('Spam vs Not Spam Comments')
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.bar(count_data.index, count_data.values, color=['#800000', '#508D4E'])
            ax2.set_title('Count of Spam vs Not Spam Comments')
            ax2.set_ylabel('Number of Comments')
            st.pyplot(fig2)

        # Row 2: WordCloud | Time Series
        col3, col4 = st.columns(2)
        with col3:
            spam_text = " ".join(df[df['PREDICTION'] == 'Spam']['CONTENT'].dropna())
            if spam_text.strip() == "":
                st.write("No spam comments to generate WordCloud.")
            else:
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(spam_text)
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                ax3.imshow(wordcloud, interpolation='bilinear')
                ax3.axis('off')
                st.pyplot(fig3)

        with col4:
            st.subheader("📈 Spam Comments Over Time")
            plot_time_series(df)


        st.subheader("👤 Top Spam Commenters")
        top_authors = df[df['PREDICTION'] == 'Spam'].groupby('AUTHOR').size().sort_values(ascending=False).head(10)
        if top_authors.empty:
            st.write("No spam comments to show top commenters.")
        else:
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            top_authors.plot(kind='barh', color='crimson', ax=ax4)
            ax4.invert_yaxis()
            ax4.set_xlabel('Number of Spam Comments')
            st.pyplot(fig4)


# --- TAB 2: Sentiment Analyzer ---
with tab2:
    st.subheader("😊 Sentiment Analyzer")

    if url and video_id:
        df, texts = get_comments(video_id)
        transformed_texts = sentiment_vectorizer.transform(texts)
        predictions = sentiment_model.predict(transformed_texts.toarray())

        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        prediction_labels = [label_map[p] for p in predictions]
        df['PREDICTION'] = prediction_labels

        st.dataframe(df)

        count_data = df['PREDICTION'].value_counts()

        # Row 1: Pie chart | Bar chart
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(
                count_data,
                labels=count_data.index,
                startangle=90,
                shadow=True,
                autopct='%.2f%%',
                colors=['#00BFFF', '#FF6347', '#90EE90'],  # blue, tomato, lightgreen for neg, neu, pos
                wedgeprops=dict(alpha=0.8, edgecolor='#6C48C5'),
                textprops={'fontsize': 12}
            )
            centre_circle = plt.Circle((0, 0), 0.10, fc='white', linewidth=1.25, edgecolor='black')
            fig.gca().add_artist(centre_circle)
            plt.axis('equal')
            plt.title('Sentiment Distribution')
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.bar(count_data.index, count_data.values, color=['#00BFFF', '#FF6347', '#90EE90'])
            ax2.set_title('Sentiment Counts')
            ax2.set_ylabel('Number of Comments')
            st.pyplot(fig2)

        # Row 2: WordCloud | Time Series
        col3, col4 = st.columns(2)
        with col3:
            positive_text = " ".join(df[df['PREDICTION'] == 'Positive']['CONTENT'].dropna())
            if positive_text.strip() == "":
                st.write("No positive comments to generate WordCloud.")
            else:
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                ax3.imshow(wordcloud, interpolation='bilinear')
                ax3.axis('off')
                st.pyplot(fig3)

        with col4:
            st.subheader("📈 Sentiment Comments Over Time")
            plot_time_series(df)


        st.subheader("👤 Top Positive Commenters")
        top_authors = df[df['PREDICTION'] == 'Positive'].groupby('AUTHOR').size().sort_values(ascending=False).head(10)
        if top_authors.empty:
            st.write("No positive comments to show top commenters.")
        else:
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            top_authors.plot(kind='barh', color='green', ax=ax4)
            ax4.invert_yaxis()
            ax4.set_xlabel('Number of Positive Comments')
            st.pyplot(fig4)


# --- TAB 3: Toxicity Analyzer ---
with tab3:
    st.subheader("☢️ Toxicity Analyzer")

    if url and video_id:
        df, texts = get_comments(video_id)
        vect_texts = vectorizer.transform(texts)
        preds = toxic_model.predict(vect_texts.toarray())
        # Assuming threshold 0.5 for toxicity
        prediction_labels = ['Toxic' if p >= 0.5 else 'Non-Toxic' for p in preds.flatten()]
        df['PREDICTION'] = prediction_labels

        st.dataframe(df)

        count_data = df['PREDICTION'].value_counts()

        # Row 1: Pie chart | Bar chart
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(
                count_data,
                labels=count_data.index,
                startangle=90,
                shadow=True,
                autopct='%.2f%%',
                colors=['#FF4500', '#00CED1'],  # orange-red and dark turquoise
                wedgeprops=dict(alpha=0.8, edgecolor='#6C48C5'),
                textprops={'fontsize': 12}
            )
            centre_circle = plt.Circle((0, 0), 0.10, fc='white', linewidth=1.25, edgecolor='black')
            fig.gca().add_artist(centre_circle)
            plt.axis('equal')
            plt.title('Toxic vs Non-Toxic Comments')
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.bar(count_data.index, count_data.values, color=['#FF4500', '#00CED1'])
            ax2.set_title('Count of Toxic vs Non-Toxic Comments')
            ax2.set_ylabel('Number of Comments')
            st.pyplot(fig2)

        # Row 2: WordCloud | Time Series
        col3, col4 = st.columns(2)
        with col3:
            toxic_text = " ".join(df[df['PREDICTION'] == 'Toxic']['CONTENT'].dropna())
            if toxic_text.strip() == "":
                st.write("No toxic comments to generate WordCloud.")
            else:
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(toxic_text)
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                ax3.imshow(wordcloud, interpolation='bilinear')
                ax3.axis('off')
                st.pyplot(fig3)

        with col4:
            st.subheader("📈 Toxic Comments Over Time")
            plot_time_series(df)

        st.subheader("👤 Top Toxic Commenters")
        top_authors = df[df['PREDICTION'] == 'Toxic'].groupby('AUTHOR').size().sort_values(ascending=False).head(10)
        if top_authors.empty:
            st.write("No toxic comments to show top commenters.")
        else:
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            top_authors.plot(kind='barh', color='darkred', ax=ax4)
            ax4.invert_yaxis()
            ax4.set_xlabel('Number of Toxic Comments')
            st.pyplot(fig4)
