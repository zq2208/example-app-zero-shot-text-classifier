!pip install transformers torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BartForConditionalGeneration, BartTokenizer
import torch
from scipy.special import softmax
import pandas as pd
import numpy as np
import nltk
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from ntscraper import Nitter
import re
import string
from collections import Counter
import ast
import os
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora, models
import plotly.express as px
from datetime import datetime
import plotly.graph_objs as go
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
import warnings

st.set_page_config(page_title="Sentiment Analysis Web App", layout="wide")
# warnings.filterwarnings("ignore", message="st.cache is deprecated. Please use one of Streamlit's new caching commands, st.cache_data or st.cache_resource.")
# @st.cache(allow_output_mutation=True)
def load_models():
    model_v4 = AutoModelForSequenceClassification.from_pretrained("v4Fine-Tuned XLMT")

    model_v5 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    model_v1 = AutoModelForSequenceClassification.from_pretrained("Fine-Tuned XLMT")

    return model_v1, model_v4, model_v5

model_v1, model_v4, model_v5 = load_models()

# @st.cache(allow_output_mutation=True)
def load_tokenizers():
    tokenizer_v4 = AutoTokenizer.from_pretrained("v4Fine-Tuned XLMT")

    tokenizer_v5 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    tokenizer_v1 = AutoTokenizer.from_pretrained("Fine-Tuned XLMT")

    return tokenizer_v1, tokenizer_v4, tokenizer_v5

tokenizer_v1, tokenizer_v4, tokenizer_v5 = load_tokenizers()

bart_model_path = 'facebook/bart-large-cnn'

def load_bart_model():
    model = BartForConditionalGeneration.from_pretrained(bart_model_path)
    tokenizer = BartTokenizer.from_pretrained(bart_model_path)
    return model, tokenizer

#------------------------------Preprocessing Functions-------------------------------------------------------------------------------------------#
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define your label mapping for adjusting labels from -1/1 to 0/1
label_map = {-1: 0, 1: 1}

# Function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text)

# Function to convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()

# Function to remove punctuations
def remove_punctuations(text, exclude="-!"):
    punctuations = set(string.punctuation) - set(exclude)
    return ''.join(char for char in text if char not in punctuations)

# Function to remove irregular spaces
def remove_irregular_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

# Function to replace slang/OOV words
def replace_slang(text, oov_dict):
    return ' '.join([oov_dict.get(word, word) for word in text.split()])

# Function to remove stopwords
def remove_stopwords(text, stopwords_list):
    return ' '.join([word for word in text.split() if word not in stopwords_list])

# Function to segment Chinese words
def segment_chinese(text):
    return ' '.join(jieba.cut(text))

# Function to remove rare words
def remove_rare_words(text, rare_words_set):
    return ' '.join([word for word in text.split() if word not in rare_words_set])

# Load OOV dictionary
oov_dict = {}
with open('Text Files\OOV Dictionary.txt', 'r') as file:
    for line in file:
        try:
            line_dict = ast.literal_eval("{" + line.strip() + "}")
            oov_dict.update(line_dict)
        except (ValueError, SyntaxError):
            print(f"Skipping invalid line: {line.strip()}")

# Load stopwords
english_stopwords = stopwords.words('english')
mandarin_stopwords = stopwords.words('chinese')
with open('Text Files\stopwords-ms.txt', 'r', encoding='utf-8') as file:
    malay_stopwords = [line.strip() for line in file]
all_stopwords = english_stopwords + mandarin_stopwords + malay_stopwords

def preprocess_for_v1(text):
    text = remove_urls(text)
    text = convert_to_lowercase(text)
    text = remove_punctuations(text)
    text = remove_irregular_spaces(text)
    text = replace_slang(text, oov_dict)
    text = remove_stopwords(text, all_stopwords)
    return text

# V4 Preprocessing: Remove URL, Remove Irregular Spaces, Handle OOV
def preprocess_for_v4(text):
    text = remove_urls(text)
    text = remove_irregular_spaces(text)
    text = replace_slang(text, oov_dict)
    return text
#----------------------------------------------------------------------------------------------------------------------------------------------#

#------------------------------Prediction  Functions-------------------------------------------------------------------------------------------#
# Define maximum sequence length
max_length = 512

def predict_v1(text):
    inputs = tokenizer_v1(text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model_v1(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

def predict_v4(text):
    inputs = tokenizer_v4(text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model_v4(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

def predict_v5(text):
    # Encode the text using tokenizer
    encoded_input = tokenizer_v5(text, return_tensors='pt', max_length=512, truncation=True)

    # Get model output and apply softmax to get probabilities
    output = model_v5(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Map scores to labels and print them
    scores_dict = {config.id2label[i]: score for i, score in enumerate(scores)}
    print("Scores:", scores_dict)  # Print the scores for debugging

    return scores_dict

def ensemble_decision(v1_pred, v4_pred, v5_scores):
    # Define your label mapping for V5
    label_map = {"negative": 0, "positive": 1, "neutral": "neutral"}
    
    # Get the label with the highest score from V5
    v5_label = max(v5_scores, key=v5_scores.get)
    v5_mapped_label = label_map.get(v5_label, "neutral")
    
    if v5_mapped_label == "neutral":
        # If V5 is neutral, decide based on V1 and V4
        if v1_pred == v4_pred:
            return v1_pred  # Both agree
        else:
            # If V1 and V4 disagree, use the second-highest score from V5 as a tiebreaker
            sorted_scores = sorted(v5_scores.items(), key=lambda item: item[1], reverse=True)
            # Get the label with the second-highest score (sorted_scores[1])
            second_highest_label, _ = sorted_scores[1]
            return label_map.get(second_highest_label, 0)  # Default to 0 if label is not found
    else:
        # If V5 is not neutral, proceed with normal majority voting
        predictions = [v1_pred, v4_pred, v5_mapped_label]
        return max(set(predictions), key=predictions.count)  # Majority voting

def sentiment_analysis_pipeline(text):
    try:
        # Preprocess the text for each model
        preprocessed_text_v1 = preprocess_for_v1(text)
        preprocessed_text_v4 = preprocess_for_v4(text)

        # Get predictions from each model
        v1_pred = predict_v1(preprocessed_text_v1)
        v4_pred = predict_v4(preprocessed_text_v4)
        v5_scores = predict_v5(preprocessed_text_v4)  # Get all scores from V5

        # Make the final decision based on the ensemble
        final_decision = ensemble_decision(v1_pred, v4_pred, v5_scores)

        # Return all relevant information
        return {
            "final_decision": final_decision,
            "v1_prediction": v1_pred,
            "v4_prediction": v4_pred,
            "v5_scores": v5_scores
        }
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return None
#----------------------------------------------------------------------------------------------------------------------------------------------#

#------------------------------Web Scraping Functions-------------------------------------------------------------------------------------------#
def analyze_sentiment(text):
    # Preprocess the text for each model
    preprocessed_text_v1 = preprocess_for_v1(text)
    preprocessed_text_v4 = preprocess_for_v4(text)

    # Get predictions from each model
    # These functions need to be defined; they should return the predicted sentiment label or score
    v1_pred = predict_v1(preprocessed_text_v1)
    v4_pred = predict_v4(preprocessed_text_v4)
    v5_scores = predict_v5(preprocessed_text_v4)  # Get all scores from V5

    # Use the ensemble decision function to determine the final sentiment
    final_decision = ensemble_decision(v1_pred, v4_pred, v5_scores)

    return final_decision

# Initialize the scraper
scraper = Nitter(0)

def get_tweets(name, modes, no):
    """
    Fetch tweets by a given name, mode, and number.

    :param name: Twitter handle or hashtag.
    :param modes: 'user' for user timeline or 'hashtag' for hashtag search.
    :param no: Number of tweets to fetch.
    :return: DataFrame containing tweet details and sentiment.
    """
    tweets = scraper.get_tweets(name, mode=modes, number=no)
    final_tweets = []

    for x in tweets['tweets']:
        data = {
            'text': x['text'],
            'date': x['date'],
            'likes': x['stats']['likes'],
            'comments': x['stats']['comments'],
            'sentiment': analyze_sentiment(x['text'])  # Analyze the sentiment of the tweet
        }
        final_tweets.append(data)

    return pd.DataFrame(final_tweets)
#----------------------------------------------------------------------------------------------------------------------------------------------#

def main():
    st.title("Bahasa Rojak Sentiment Analysis with XLMR 🤗😒")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Homepage 🏠", "Text Predictions 📖", "Web Scraping 𝕏"])

    with tab1:
        st.header("About This Project")
        st.markdown(
        """
        ## Welcome to the Sentiment Analysis Project! 🎉
        
        ### About This Project
        
        This sentiment analysis project is designed to provide a comprehensive understanding 
        of public opinion on various topics through the analysis of social media posts. 
        It utilizes an ensemble of advanced machine learning models to ensure accurate and real-time sentiment analysis:
        
        - **XLM-T**: A multilingual XLM-roBERTa-base model trained on ~198M tweets and 
        finetuned for sentiment analysis. 
        [Read the paper here.](https://doi.org/10.48550/arXiv.2104.12250)
        
        - **V1**: Further Fine-Tuned version of XLMT on a labelled Bahasa Rojak dataset to 
        enhance prediction accuracy for targeted sentiment contexts.
        
        - **V4**: Similar to V1 in terms of the data used for training but with 
        far fewer preprocessing steps (e.g., maintaining stopwords, etc.).
        
        These models work together to analyze text inputs or fetched tweets, providing 
        insights into public opinion and emotional tone. 
        Whether you're interested in understanding the sentiment behind social media posts, news articles, 
        or any other form of text, this tool is here to help.
        
        ### How to Use
        
        1. **Analyze Text**: Input any text into the provided text area to analyze its sentiment instantly.
        
        2. **Twitter Sentiment Analysis**: Enter a Twitter handle or hashtag to fetch and analyze tweets.
        The system will provide a comprehensive sentiment analysis, including individual model predictions 
        and interactive visualizations.
        
        ### Dive In!
        
        Ready to explore what the public is feeling? Scroll down to input your text or a Twitter handle/hashtag 
        and discover the sentiments being expressed. Have fun exploring and analyzing!
        """
    )
    
    with tab2:
        st.header("Predict Text Sentiment ➕➖")
        # Text sentiment analysis section
        user_input = st.text_area("Enter Text for Sentiment Analysis", "")
        if st.button("Analyze Sentiment"):
            if user_input:
                if user_input:
                    processed_text = preprocess_for_v1(user_input)

                    summarize_text(user_input)

                    perform_lda(processed_text)

                    analysis_results = sentiment_analysis_pipeline(user_input)
                    if analysis_results:
                         display_analysis_results(analysis_results)

    with tab3:
        st.header("Twitter Sentiment Analysis 𝕏")
        st.markdown("Analyze any topics and user tweets. (Eg. #MalaysiaMadani / anwaribrahim)")

        # User inputs for Twitter handle/hashtag and number of tweets
        query = st.text_input("Enter the Twitter handle or hashtag", "#MalaysiaMadani")
        num_tweets = st.slider("Number of tweets to analyze", 5, 800, 10)
        mode_selection = st.radio("Choose the type of tweets to fetch", ['hashtag','user'])

        # Button to fetch and analyze tweets
        if st.button("Scrape and Analyze Tweets"):
            try:
                if query:
                    data = get_tweets(query, mode_selection, num_tweets)
                    if not data.empty:
                        # Display data and sentiment analysis
                        st.write(data)
                        display_sentiment_and_statistics(data)
                        plot_sentiment_over_time(data)
                        plot_wordcloud(data)
                    else:
                        st.warning("No tweets found.")
                else:
                    st.warning("Please enter a valid Twitter handle or hashtag.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Footer or additional information
    st.write("---")
    st.markdown("Developed by [Boon Yong Yeow](https://github.com/Bernardbyy), Ong Sheng Hao, and Ong Ker Jing")

def perform_lda(text_data):
    # Tokenize the text data
    tokens = text_data.split()

    # Create a Gensim Dictionary from the tokens
    dictionary = corpora.Dictionary([tokens])

    # Create a Gensim corpus from the tokens
    corpus = [dictionary.doc2bow(tokens)]

    # Create an LDA model using the corpus
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    # Get the topics
    topics = lda_model.print_topics(num_words=4)

    # Display the topics
    cols = st.columns(len(topics))
    
    for idx, (col, topic) in enumerate(zip(cols, topics)):
        with col:
            # Extract topic number and keywords
            topic_num, keywords = topic
            st.markdown(f"**Topic {topic_num + 1}**")
            st.markdown(
                f"""
                <div style='
                    text-align: left;
                    background-color: white;
                    padding: 10px;
                    border-radius: 10px;
                    color: #2E86C1;  # Change this for different text colors
                    font-family: "Arial";  # Change this for different fonts
                    font-size: 16px;  # Change this for different font sizes
                '>
                    {keywords}
                </div>
                """, 
                unsafe_allow_html=True
            )

def summarize_text(text):
    model, tokenizer = load_bart_model()
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Display the summary
    st.subheader('Text Summary')
    st.write(summary)

def display_analysis_results(analysis_results):
    sentiment_map = {0: "Negative 😞", 1: "Positive 😊", "neutral": "Neutral 😐"}

    # Container for all models' predictions
    st.markdown("<div style='display: flex; flex-wrap: wrap; justify-content: space-around; gap: 20px;'>", unsafe_allow_html=True)

    # Display individual model predictions in boxes
    for model_name, model_pred in [("V1", 'v1_prediction'), ("V4", 'v4_prediction')]:
        model_prediction_text = sentiment_map.get(analysis_results[model_pred], "Unknown")
        model_prediction_color = "green" if analysis_results[model_pred] == 1 else "red" if analysis_results[model_pred] == 0 else "#808080"  # Gray for neutral/unknown
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; border: 1px solid #ccc; text-align: center;'>
                <h4>{model_name} Prediction</h4>
                <p style='color: {model_prediction_color}; font-size: 18px; font-weight: bold;'>{model_prediction_text}</p>
            </div>
        """, unsafe_allow_html=True)

    # Display V5 scores
    st.write("### Model V5 Scores")
    for label, score in analysis_results['v5_scores'].items():
        score_color = "green" if label == 'positive' else "red" if label == 'negative' else "#808080"
        st.markdown(f"""
            <div style='padding: 10px; border-radius: 10px; border: 1px solid #ccc; margin: 10px 0;'>
                <p><b>{label.capitalize()}:</b> <span style='color: {score_color};'>{score:.4f}</span></p>
            </div>
        """, unsafe_allow_html=True)

    # Display the final decision
    final_decision_text = sentiment_map.get(analysis_results['final_decision'], "Unknown")
    final_decision_color = "green" if analysis_results['final_decision'] == 1 else "red" if analysis_results['final_decision'] == 0 else "#808080"
    st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; border: 2px solid {final_decision_color}; text-align: center;'>
            <h4>Final Decision</h4>
            <p style='color: {final_decision_color}; font-size: 20px; font-weight: bold;'>{final_decision_text}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Close the container

def parse_date(date_str):
    # Remove the '·' and 'UTC' parts from the date string
    date_str = date_str.replace('·', '').replace('UTC', '').strip()
    # Define the date format
    date_format = '%b %d, %Y %I:%M %p'
    # Parse the date
    return datetime.strptime(date_str, date_format)

def plot_sentiment_over_time(data):
    # Convert 'date' column to datetime objects using the custom parser
    data['date'] = data['date'].apply(parse_date)

    # Resample the data to a regular frequency, counting the number of positive and negative sentiments
    sentiment_counts = data.groupby([pd.Grouper(key='date', freq='H'), 'sentiment']).size().unstack().fillna(0)

    # Create traces for the interactive plot
    traces = []
    if 0 in sentiment_counts.columns:
        traces.append(go.Scatter(
            x=sentiment_counts.index,
            y=sentiment_counts[0],
            mode='lines',
            name='Negative Sentiment',
            line=dict(color='red'),
        ))
    if 1 in sentiment_counts.columns:
        traces.append(go.Scatter(
            x=sentiment_counts.index,
            y=sentiment_counts[1],
            mode='lines',
            name='Positive Sentiment',
            line=dict(color='green'),
        ))

    # Define layout options
    layout = go.Layout(
        title='Sentiment Over Time',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title='Number of Tweets'),
        hovermode='closest',
        # Set the width and height of the figure (1.5 times bigger)
        width=900 * 1.45,  # Adjust the width as needed
        height=600 * 1.35,  # Adjust the height as needed
    )

    # Create the figure with the traces and layout
    fig = go.Figure(data=traces, layout=layout)

    # Center the plot (Streamlit usually centers automatically, this is just to ensure)
    st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

def display_sentiment_and_statistics(data):
    # Create two columns for the layout
    col1, col2 = st.columns(2)
    
    # First column for statistics
    with col1:
        st.write("### Tweet Statistics 📊")
        
        # Define styles for the boxes
        box_style = """
        <style>
            .stat-box {
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
                background-color: white;
            }
            .stat-label {
                color: black;
            }
            .stat-number {
                color: #1DA1F2;  /* Twitter blue color */
                font-weight: bold;
            }
        </style>
        """

        # Add the style to the page
        st.markdown(box_style, unsafe_allow_html=True)

        # Number of tweets scraped
        number_of_tweets = len(data)
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-label">Number of Tweets Scraped: </span>
            <span class="stat-number">{number_of_tweets}</span>
        </div>
        """, unsafe_allow_html=True)

        # Time period of scraped tweets
        earliest_date = data['date'].min()
        latest_date = data['date'].max()
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-label">Time Period of Tweets: </span>
            <span class="stat-number">{earliest_date}</span>
            <span class="stat-label"> to </span>
            <span class="stat-number">{latest_date}</span>
        </div>
        """, unsafe_allow_html=True)

        # Average length of tweets
        average_length = data['text'].apply(len).mean()
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-label">Average Length of Tweets: </span>
            <span class="stat-number">{average_length:.2f} characters</span>
        </div>
        """, unsafe_allow_html=True)

        # Average like count
        average_likes = data['likes'].mean()
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-label">Average Like Count: </span>
            <span class="stat-number">{average_likes:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Average comments count
        average_comments = data['comments'].mean()
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-label">Average Comment Count: </span>
            <span class="stat-number">{average_comments:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Second column for sentiment distribution pie chart
    with col2:
        st.write("### Sentiment Proportions 🥧")
        # Assuming 'sentiment' column has 0s and 1s
        sentiment_counts = data['sentiment'].value_counts(normalize=True)  # Get proportions instead of counts

        # Handle cases where there's only one class (all 0s or all 1s)
        if len(sentiment_counts) == 1:
            # If only one sentiment class, append the other class with 0 count
            if 0 not in sentiment_counts.index:
                sentiment_counts[0] = 0  # if only positive, add negative
            else:
                sentiment_counts[1] = 0  # if only negative, add positive

        # Creating a DataFrame for the Plotly pie chart
        df = pd.DataFrame({
            'Sentiment': ['Negative', 'Positive'],
            'Proportion': sentiment_counts.loc[[0, 1]]  # Ensure correct order
        })

        # Using Plotly Express to create the pie chart
        fig = px.pie(df, values='Proportion', names='Sentiment', color='Sentiment',
                     color_discrete_map={'Negative': '#ff9999', 'Positive': '#66b3ff'})

        # Adjusting layout to fit within the container
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))

        # Making the pie chart interactive
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])  # 'pull' slightly extracts the slice from the pie

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

# Function to preprocess all tweets in a dataset
def preprocess_tweets(tweets):
    preprocessed_tweets = [preprocess_for_v1(tweet) for tweet in tweets]
    return " ".join(preprocessed_tweets) 

def plot_wordcloud(data):
    # Separate tweets based on sentiment
    positive_tweets = data[data['sentiment'] == 1]['text'].tolist()
    negative_tweets = data[data['sentiment'] == 0]['text'].tolist()
    
    # Set Streamlit option to suppress the Pyplot global use warning
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Initialize subplots with a background color
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0E1113')  # Change to a desired background color

    # Create and display word cloud for positive tweets
    if positive_tweets:
        positive_text = preprocess_tweets(positive_tweets)
        positive_wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white").generate(positive_text)
        axes[0].imshow(positive_wordcloud, interpolation='bilinear')
        axes[0].set_title('Positive Sentiment Word Cloud', color='white', fontsize=18, fontweight='bold')
        axes[0].axis("off")  # Turn off the axis

    # Create and display word cloud for negative tweets
    if negative_tweets:
        negative_text = preprocess_tweets(negative_tweets)
        negative_wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white").generate(negative_text)
        axes[1].imshow(negative_wordcloud, interpolation='bilinear')
        axes[1].set_title('Negative Sentiment Word Cloud', color='white', fontsize=18, fontweight='bold')
        axes[1].axis("off")  # Turn off the axis

    # Adjust the layout with a bit more space and a contrasting background
    plt.subplots_adjust(wspace=0.1)  # Adjust the space between the plots
    
    # Display the plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    nltk.download('stopwords')
    main()
