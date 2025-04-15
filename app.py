import streamlit as st
import os
import json
import re
import io
from firecrawl import FirecrawlApp
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd 
from datetime import datetime 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from typing import List, Dict, Any, Optional
import unicodedata
import emoji
import dateparser

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

if not firecrawl_api_key:
    st.error("Error: FIRECRAWL_API_KEY is missing")
    st.stop()

app = FirecrawlApp(api_key=firecrawl_api_key)

BASE_URLS = {
    "Amazon": "https://www.amazon.in/s?k=",
    "Flipkart": "https://www.flipkart.com/search?q=",
    "Myntra": "https://www.myntra.com/"
}

# New functions for review parsing
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = emoji.replace_emoji(text, replace='')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.strip()

def extract_rating(text: str) -> Optional[float]:
    """Extract rating from text"""
    star_matches = re.findall(r'★{1,5}', text)
    numeric_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:out of 5|/5)', text)
    if star_matches:
        return len(star_matches[0])
    elif numeric_matches:
        return float(numeric_matches[0])
    return None

def detect_sentiment(text: str) -> Dict[str, float]:
    """Advanced sentiment detection using contextual analysis"""
    positive_markers = ['great', 'amazing', 'excellent', 'love', 'fantastic', 'recommend', 'perfect', 'awesome', 'best']
    negative_markers = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed', 'poor', 'useless', 'fail']
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_markers if word in text_lower)
    negative_count = sum(1 for word in negative_markers if word in text_lower)
    total_markers = positive_count + negative_count
    if total_markers == 0:
        return {"sentiment": "neutral", "sentiment_score": 0.5}
    sentiment_score = positive_count / total_markers
    sentiment = (
        "positive" if sentiment_score > 0.6 else
        "negative" if sentiment_score < 0.4 else
        "neutral"
    )
    return {
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 2)
    }

def parse_date(date_str: str) -> Optional[str]:
    """Parse various date formats, handle relative dates"""
    try:
        parsed_date = dateparser.parse(date_str, settings={'RELATIVE_BASE': datetime.now()})
        return parsed_date.strftime('%Y-%m-%d') if parsed_date else None
    except:
        return None

def is_verified_purchase(text: str) -> bool:
    """Detect verified purchase status"""
    verification_phrases = ['verified purchase', 'confirmed buyer', 'purchase verified', 'confirmed purchase']
    return any(phrase in text.lower() for phrase in verification_phrases)

def extract_key_phrases(text: str) -> List[str]:
    """Extract meaningful phrases"""
    patterns = [
        r'\b\w+ is (great|amazing|terrible|bad)\b',
        r'(love|hate) (how|that)',
        r'(best|worst) \w+ ever'
    ]
    key_phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_phrases.extend([' '.join(match) for match in matches])
    return key_phrases[:3]

def parse_reviews(raw_text: str) -> List[Dict[str, Any]]:
    """Parse unstructured review text into structured review objects"""
    review_pattern = re.compile(
        r'(?P<reviewer>[\w\s]+)\s*'  # Reviewer name
        r'(?P<rating>★{1,5}|\d+(?:\.\d+)?(?:/5)?)\s*'  # Rating
        r'(?P<verified>Verified Purchase)?\s*\|\s*'  # Verification
        r'(?P<date>[\w\s,]+)\n'  # Date
        r'(?P<text>.*?)(?=\n\n|\Z)',  # Review text
        re.DOTALL | re.MULTILINE
    )
    parsed_reviews = []
    for match in review_pattern.finditer(raw_text):
        try:
            reviewer_name = clean_text(match.group('reviewer'))
            rating = extract_rating(match.group('rating'))
            verified = match.group('verified') is not None
            date_str = match.group('date')
            review_text = clean_text(match.group('text'))
            sentiment_data = detect_sentiment(review_text)
            review = {
                "rating": rating,
                "review_date": parse_date(date_str),
                "review_text": review_text,
                "reviewer_name": reviewer_name,
                "verified_purchase": verified,
                "sentiment": sentiment_data["sentiment"],
                "sentiment_score": sentiment_data["sentiment_score"],
                "key_phrases": extract_key_phrases(review_text)
            }
            parsed_reviews.append(review)
        except Exception as e:
            print(f"Error parsing review: {e}")
    return parsed_reviews

# Example usage of the new parsing function
def parse_reviews_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse reviews from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return parse_reviews(raw_text)

# Integrate the new parsing function into the existing workflow
def extract_reviews_details(crawl_data):
    """Extract detailed review information using Gemini AI"""
    try:
        print("Extracting review details with Gemini...")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = (
            "You are a review extraction assistant. Extract all customer reviews from the scraped web data "
            "and return them in a valid JSON array. Each review should be a JSON object with these fields:\n"
            "- Rating (number between 1-5)\n"
            "- Review Date (string)\n"
            "- Review Title (string)\n"
            "- Review Content (string)\n"
            "- Reviewer Name (string)\n"
            "- Verified Purchase (boolean)\n\n"
            "Format the response as a valid JSON array of objects. Include ONLY the JSON data, no explanations.\n\n"
            "Here's the data to process:\n"
            f"{json.dumps(crawl_data, indent=2)}"
        )
        response = model.generate_content(prompt)
        
        if not response or not response.text.strip():
            print("Error: Received empty response from Generative AI.")
            return None
            
        response_text = response.text.strip()
        if response_text.startswith("```"):
            blocks = response_text.split("```")
            for block in blocks:
                if block.strip().startswith("[") or block.strip().startswith("{"):
                    response_text = block.strip()
                    break
        
        structured_reviews = json.loads(repair_json(response_text))
        
        # Ensure we have a list of reviews
        if not isinstance(structured_reviews, list):
            structured_reviews = [structured_reviews]
            
        # Add sentiment analysis to each review
        for review in structured_reviews:
            if 'Review Content' in review:
                content = review['Review Content']
            else:
                content = ""
                
            sentiment = analyze_sentiment(content)
            review['Sentiment'] = sentiment['sentiment']
            review['Sentiment Score'] = sentiment['compound']
        
        return structured_reviews
        
    except Exception as e:
        print(f"Error extracting reviews: {e}")
        return None

# The rest of the existing code remains unchanged
