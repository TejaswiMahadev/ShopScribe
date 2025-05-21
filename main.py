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
import seaborn as sns 
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



def generate_search_url(website, query):
    """Generate search URL for the specified website"""
    base_url = BASE_URLS.get(website)
    if not base_url:
        print(f"Error: Website '{website}' not supported.")
        return None
    if website == "Myntra":
        return f"{base_url}{query.replace(' ', '%20')}"
    else:
        return f"{base_url}{query.replace(' ', '+')}"
    
def crawl_website(url):
    """Crawl the specified URL using Firecrawl"""
    try:
        print(f"Crawling website: {url}")
        crawl_data = app.scrape_url(url)
        print("Crawl successful. Data obtained.")
        return crawl_data
    except Exception as e:
        print(f"Error crawling website: {e}")
        return None


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
    """Improved date parsing with multiple fallbacks"""
    try:
        parsed_date = dateparser.parse(date_str, settings={
            'RELATIVE_BASE': datetime.now(),
            'PREFER_DAY_OF_MONTH': 'first',
            'DATE_ORDER': 'DMY'
        })
        
        # Fallback patterns
        if not parsed_date:
            for pattern in [
                r'\b(\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'\b(\d{1,2}/\d{1,2}/\d{4})\b'
            ]:
                match = re.search(pattern, date_str)
                if match:
                    parsed_date = pd.to_datetime(match.group(1), errors='coerce')
                    break
                    
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

def parse_date(date_str: str) -> Optional[str]:
    """Robust date parsing with multiple fallback strategies"""
    try:
        # Clean input string
        clean_str = re.sub(r'\b(Posted on|Reviewed in|on)\b', '', date_str, flags=re.IGNORECASE)
        
        # Try multiple parsers
        parsed_date = dateparser.parse(clean_str, settings={
            'RELATIVE_BASE': datetime.now(),
            'PREFER_DAY_OF_MONTH': 'first',
            'DATE_ORDER': 'DMY',
            'STRICT_PARSING': True
        })
        
        # Fallback patterns
        if not parsed_date:
            patterns = [
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # DD-MM-YYYY
                r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',      # YYYY-MM-DD
                r'\b(\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})\b',
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, clean_str)
                if match:
                    parsed_date = pd.to_datetime(match.group(0), errors='coerce')
                    if pd.notnull(parsed_date):
                        break
        
        return parsed_date.strftime('%Y-%m-%d') if parsed_date else None
    except:
        return None
# Example usage of the new parsing function
def parse_reviews_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse reviews from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return parse_reviews(raw_text)


def process_reviews_for_analysis(reviews):
    """
    Transform raw reviews into a standardized format for sentiment analysis
    
    Args:
        reviews (list): Raw reviews from crawl data
    
    Returns:
        list: Processed reviews with consistent structure
    """
    processed_reviews = []
    
    # Handle case where reviews is not a list
    if not isinstance(reviews, list):
        if isinstance(reviews, str):
            reviews = [{"Review Content": reviews}]
        elif isinstance(reviews, dict):
            reviews = [reviews]
        else:
            print(f"Invalid reviews format: {type(reviews)}")
            return []
    
    for review in reviews:
        try:
            # Handle different input types
            if isinstance(review, str):
                # If it's just a string, create a minimal review object
                review_content = review
                review = {"Review Content": review_content}
            
            if not isinstance(review, dict):
                print(f"Skipping invalid review: {type(review)}")
                continue
            
            # Create a standardized review
            processed_review = {
                "Review Content": review.get("Review Content", ""),
                "Sentiment": "neutral",
                "Sentiment Score": 0,
                "Review Date": review.get("Review Date", 
                              review.get("Date", 
                              datetime.now().strftime("%Y-%m-%d"))),
                "Rating": review.get("Rating", None)
            }
            
            # Add sentiment analysis
            if processed_review["Review Content"]:
                sentiment_result = analyze_sentiment(processed_review["Review Content"])
                processed_review["Sentiment"] = sentiment_result['sentiment']
                processed_review["Sentiment Score"] = round(sentiment_result['compound'], 2)
            
            # Add rating if available
            if processed_review["Rating"] is None:
                # Try to extract rating from Review Content or other sources
                rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of 5|/5)', processed_review["Review Content"])
                if rating_match:
                    processed_review["Rating"] = float(rating_match.group(1))
                elif "rating" in review:
                    try:
                        processed_review["Rating"] = float(review["rating"])
                    except (ValueError, TypeError):
                        pass
            
            processed_reviews.append(processed_review)
        except Exception as e:
            print(f"Error processing review: {e}")
            continue
    
    return processed_reviews


# Integrate the new parsing function into the existing workflow
def extract_reviews_details(crawl_data):
    """Enhanced review extraction with robust processing"""
    try:
        print("Extracting review details with Gemini...")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = (
            "You are a review extraction assistant. Extract all customer reviews from the scraped web data "
            "and return them in a valid JSON array. Each review should be a JSON object with these fields:\n"
            "- Rating (number between 1-5)\n"
            "- Review Date (string, preferably in YYYY-MM-DD format)\n"
            "- Review Title (optional string)\n"
            "- Review Content (string with the full review text)\n"
            "- Reviewer Name (optional string)\n"
            "- Verified Purchase (optional boolean)\n\n"
            "Format the response as a valid JSON array of objects. Return ONLY the JSON data.\n\n"
            "Here's the data to process:\n"
            f"{json.dumps(crawl_data, indent=2)}"
        )
        response = model.generate_content(prompt)
        
        if not response or not response.text.strip():
            print("Error: Received empty response from Generative AI.")
            return []
        
        response_text = response.text.strip()
        # Extract JSON from markdown/code blocks if necessary
        if response_text.startswith("```"):
            blocks = response_text.split("```")
            for block in blocks:
                if block.strip().startswith("[") or block.strip().startswith("{"):
                    response_text = block.strip()
                    break
        
        # Repair JSON
        response_text = repair_json(response_text)
        
        try:
            structured_reviews = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Problematic response text: {response_text}")
            # Try to extract individual reviews if full JSON parsing fails
            reviews = []
            review_matches = re.finditer(r'{[^{}]*}', response_text)
            for match in review_matches:
                try:
                    review_json = match.group()
                    review = json.loads(review_json)
                    reviews.append(review)
                except:
                    continue
            
            if reviews:
                structured_reviews = reviews
            else:
                # Last resort: create a single review with the raw text
                structured_reviews = [{"Review Content": "Failed to parse reviews properly"}]
        
        # Ensure we have a list of reviews
        if not isinstance(structured_reviews, list):
            structured_reviews = [structured_reviews]
        
        # Process reviews for sentiment analysis
        processed_reviews = process_reviews_for_analysis(structured_reviews)
        
        print(f"Successfully processed {len(processed_reviews)} reviews")
        return processed_reviews
        
    except Exception as e:
        print(f"Error extracting reviews: {e}")
        import traceback
        traceback.print_exc()
        return []

def clean_price(price_str):
    """Convert price string to clean format"""
    if not price_str:
        return None
   
    price_str = price_str.replace('\\u20b9', '₹')
   
    numbers = re.findall(r'[\d,]+\.?\d*', price_str)
    if numbers:
       
        return '₹' + numbers[0].replace(',', '')
    return None

def extract_rating(rating_str):
    """Extract numeric rating from string like '4.0 out of 5 stars'"""
    if not rating_str:
        return None
    match = re.search(r'(\d+\.?\d*)\s*out of\s*\d+\s*stars?', rating_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    try:
        return float(rating_str)
    except ValueError:
        return None

def extract_reviews(reviews_str):
    """Extract number of reviews from string and handle thousands separators"""
    if not reviews_str:
        return None
    reviews_str = reviews_str.replace(',', '')
    numbers = re.findall(r'\d+', reviews_str)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            return None
    return None

def repair_json(text):
    """Attempt to repair common JSON issues"""
    if not text:
        print("Warning: Empty text received")
        return "[]"
        
    text = text.strip()
    if not text.startswith('['):
        first_brace = text.find('{')
        if first_brace != -1:
            last_brace = text.rfind('}')
            if last_brace != -1:
                text = '[' + text[first_brace:last_brace + 1] + ']'
            else:
                print("Warning: No closing brace found")
                return "[]"
        else:
            print("Warning: No JSON object found in text")
            return "[]"
    text = text.replace('"', '"').replace('"', '"')  
    text = text.replace("'", '"') 
    text = text.replace('\\"', '"')
    text = text.replace('""', '"')
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)
    text = ' '.join(text.split())
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'
        
    return text

def repair_json_products(text):
    """Attempt to repair common JSON issues in the response."""
    if not text:
        print("Warning: Empty text received")
        return "[]"

    text = text.strip()
    if not text.startswith('['):
        first_brace = text.find('{')
        if first_brace != -1:
            last_brace = text.rfind('}')
            if last_brace != -1:
                text = '[' + text[first_brace:last_brace + 1] + ']'
            else:
                print("Warning: No closing brace found")
                return "[]"
        else:
            print("Warning: No JSON object found in text")
            return "[]"
    text = text.replace("‘", '"').replace("’", '"').replace("“", '"').replace("”", '"')
    text = re.sub(r"(?<!\\)'", '"', text)
    text = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)
    text = ' '.join(text.split())
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'

    return text



def validate_product(product):
    """Validate and clean a single product entry"""
    required_fields = {
        'Product Name': str,
        'Price': clean_price,
        'Description': str,
        'Rating': extract_rating,
        'Reviews': extract_reviews,
        'Review Content': str,
        'Brand': str,
        'Product URL': str
    }
    
    cleaned_product = {}
    for field, processor in required_fields.items():
        value = product.get(field)
        if value is None or value == "":
            cleaned_product[field] = None
            continue
        try:
            cleaned_product[field] = processor(value)
        except Exception as e:
            print(f"Warning: Could not process {field} value '{value}'. Setting to None. Error: {str(e)}")
            cleaned_product[field] = None
    
    return cleaned_product

def structure_data_with_gemini(crawl_data, website=None , search_query=None):
    """Structure scraped data using Gemini AI"""
    try:
        print("Structuring data with Gemini...")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = (
            "You are a data extraction assistant. Extract product information from the scraped web data "
            "and return it in a valid JSON array. Each product should be a JSON object with these fields:\n"
            "- Product Name (string)\n"
            "- Price (string with currency symbol)\n"
            "- Description (string)\n"
            "- Rating (string)\n"
            "- Reviews (string)\n"
            "- Review Content (array of strings)\n"
            "- Brand (string)\n"
            "- Image URL (string)\n"
            "- Product URL (string)\n\n"
            "Format the response as a valid JSON array of objects. Include ONLY the JSON data, no explanations or markdown.\n\n"
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
                    
        try:
            structured_data = json.loads(response_text)
            if isinstance(structured_data, list):
                return structured_data
            elif isinstance(structured_data, dict):
                return [structured_data]
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            
            # Try to salvage partial data
            product_matches = re.finditer(r'{[^{}]*}', response_text)
            products = []
            for match in product_matches:
                try:
                    product_text = match.group()
                    if not product_text.startswith("{"):
                        product_text = "{" + product_text
                    if not product_text.endswith("}"):
                        product_text = product_text + "}"
                    product = json.loads(product_text)
                    products.append(product)
                except:
                    continue
            
            if products:
                return products
            
        return None
        
    except Exception as e:
        print(f"Error structuring data: {e}")
        # Export the raw data in case of error
        return None

def crawl_product_details(product_url):
    """Crawl the product details page and structure the data using Gemini"""
    try:
        print(f"Crawling product details page: {product_url}")
        product_data = app.scrape_url(product_url)
        print("Product details crawled successfully.")
        structured_details = structure_data_with_gemini(product_data)
        
        if structured_details:
            print("Product details structured successfully.")
            # Extract reviews with sentiment analysis
            reviews_data = extract_reviews_details(product_data)
            
            # Ensure structured_details is properly formatted
            if isinstance(structured_details, list):
                main_product = structured_details[0]
            else:
                main_product = structured_details
                
            if reviews_data:
                main_product['Review Content'] = reviews_data
                
            return main_product
            
        return None
        
    except Exception as e:
        print(f"Error crawling product details: {e}")
        return None


def export_reviews_to_excel(reviews_data, product_info, website, search_query):
    """
    Export reviews data to a separate Excel file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"product_reviews_{website}_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Product Information Sheet
            product_df = pd.DataFrame({
                'Property': ['Product Name', 'Price', 'Brand', 'Overall Rating', 'Total Reviews'],
                'Value': [
                    product_info.get('Product Name', 'N/A'),
                    product_info.get('Price', 'N/A'),
                    product_info.get('Brand', 'N/A'),
                    product_info.get('Rating', 'N/A'),
                    product_info.get('Reviews', 'N/A')
                ]
            })
            product_df.to_excel(writer, sheet_name='Product Information', index=False)
            
            # Reviews Sheet
            if reviews_data:
                reviews_df = pd.DataFrame(reviews_data)
                reviews_df.to_excel(writer, sheet_name='Reviews', index=False)  # Removed sentiment columns here
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = min(adjusted_width, 50)
        
        print(f"\nReviews exported successfully to {filename}")
        return filename
    except Exception as e:
        print(f"Error during Excel export: {str(e)}")
        return None



def display_product_details(product_details):
    """Display product details in a formatted way"""
    if not product_details:
        print("No product details available.")
        return

    print("\nDetailed Product Information:")
    print("-" * 50)
    for key, value in product_details.items():
        if isinstance(value, (dict, list)):
            print(f"\n{key}:")
            print(json.dumps(value, indent=2))
        else:
            print(f"{key}: {value}")
    print("-" * 50)



def parse_reviews(raw_data):
    """Parse review data from raw text."""
    reviews = []

    # Example regex patterns (these will need to be adjusted based on your actual data)
    review_pattern = re.compile(r'Review by (.+?):\s*Rating: (\d+\.\d+) out of 5\s*Title: (.+?)\s*Content: (.+?)\s*Date: (.+?)\s*Helpful Votes: (\d+)', re.DOTALL)

    for match in review_pattern.finditer(raw_data):
        reviewer_name = match.group(1).strip()
        rating = match.group(2).strip()
        review_title = match.group(3).strip()
        review_content = match.group(4).strip()
        review_date = match.group(5).strip()
        helpful_votes = match.group(6).strip()

        review = {
            "Reviewer Name": reviewer_name,
            "Rating": rating,
            "Review Title": review_title,
            "Review Content": review_content,
            "Review Date": review_date,
            "Helpful Votes": helpful_votes
        }
        reviews.append(review)

    return reviews

def structure_data_manually(raw_data):
    """Structure data manually using custom parsing logic."""
    try:
        print("Structuring data manually...")
        structured_reviews = parse_reviews(raw_data)

        if structured_reviews:
            print(f"Successfully structured {len(structured_reviews)} reviews.")
            return structured_reviews
        else:
            print("No valid reviews found in raw data.")
            return None

    except Exception as e:
        print(f"Error structuring data: {e}")
        return None

def analyze_sentiment(text):
    """Enhanced sentiment analysis using VADER with intensity checks"""
    try:
        # Handle non-string inputs
        if not isinstance(text, str):
            if text is None:
                return {
                    'compound': 0,
                    'sentiment': 'neutral',
                    'positive': 0,
                    'negative': 0,
                    'neutral': 1
                }
            text = str(text)
        
        # Clean text for better analysis
        text = clean_text(text)
        
        # Skip empty text
        if not text:
            return {
                'compound': 0,
                'sentiment': 'neutral',
                'positive': 0,
                'negative': 0,
                'neutral': 1
            }
        
        # Perform sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        
        # Enhanced sentiment classification
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            # Check for strong neutral indicators
            if any(word in text.lower() for word in ['average', 'decent', 'okay', 'mediocre']):
                sentiment = 'neutral (mixed)'
            else:
                sentiment = 'neutral'
        
        return {
            'compound': scores['compound'],
            'sentiment': sentiment,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Return neutral sentiment as fallback
        return {
            'compound': 0,
            'sentiment': 'neutral',
            'positive': 0,
            'negative': 0,
            'neutral': 1
        }

def perform_aspect_analysis(reviews):
    """Perform aspect-based sentiment analysis using Gemini"""
    try:
        if not reviews:
            st.error("No reviews available for aspect analysis")
            return None

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = f"""
        Analyze the following product reviews and identify key aspects mentioned along with their sentiment.
        Return ONLY a valid JSON object with aspects as keys and values containing sentiment (positive/neutral/negative) 
        and example quotes. Use exactly this format:
        {{
            "aspect1": {{
                "sentiment": "positive",
                "examples": ["quote1", "quote2"]
            }},
            ...
        }}
        Reviews: {json.dumps(reviews, indent=2)}
        """
        response = model.generate_content(prompt)

        if not response or not response.text.strip():
            st.error("Aspect analysis failed: Empty response from Gemini")
            return None

        response_text = response.text.strip()
        
        # Improved JSON extraction
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Debug logging
        print("Raw response text:", response_text)

        # Validate JSON structure
        try:
            aspect_data = json.loads(response_text)
            if not isinstance(aspect_data, dict):
                st.error("Invalid aspect analysis format: Expected JSON object")
                return None
                
            # Validate aspect structure
            for aspect, details in aspect_data.items():
                if "sentiment" not in details or "examples" not in details:
                    st.error(f"Invalid aspect format for '{aspect}'")
                    return None
                if not isinstance(details["examples"], list):
                    st.error(f"Examples should be list for '{aspect}'")
                    return None
                    
            return aspect_data

        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            st.text(f"Failed to parse response: {response_text}")
            return None

    except Exception as e:
        st.error(f"Aspect analysis error: {e}")
        return None


def plot_sentiment_distribution(reviews, min_reviews=1):
    """
    Robust sentiment distribution visualization
    
    Args:
        reviews (list): List of review dictionaries
        min_reviews (int): Minimum number of reviews required to generate plot
    
    Returns:
        matplotlib.figure.Figure or None
    """
    try:
        # Validate reviews
        if not reviews or len(reviews) < min_reviews:
            print(f"Not enough reviews for distribution (need {min_reviews}, got {len(reviews) if isinstance(reviews, list) else 'not a list'})")
            return None
        
        # Prepare data with fallback mechanisms
        sentiment_data = []
        for review in reviews:
            try:
                # Handle string reviews by performing sentiment analysis
                if isinstance(review, str):
                    sentiment_result = analyze_sentiment(review)
                    sentiment = sentiment_result['sentiment']
                    score = sentiment_result['compound']
                    sentiment_data.append({
                        'Sentiment': sentiment,
                        'Score': score
                    })
                # Handle dictionary reviews
                elif isinstance(review, dict):
                    # First try to get the review content for analysis if sentiment is missing
                    if 'Sentiment' not in review and 'Review Content' in review:
                        content = review.get('Review Content', '')
                        if content and isinstance(content, str):
                            sentiment_result = analyze_sentiment(content)
                            sentiment = sentiment_result['sentiment']
                            score = sentiment_result['compound']
                        else:
                            sentiment = 'neutral'
                            score = 0
                    else:
                        # Try to get existing sentiment data
                        sentiment = review.get('Sentiment', 
                            review.get('sentiment', 
                            review.get('sent', 'neutral'))).lower()
                        
                        # Try to get sentiment score with fallbacks
                        try:
                            score = float(review.get('Sentiment Score', 
                                review.get('sentiment_score', 
                                review.get('score', 0))))
                        except (TypeError, ValueError):
                            score = 0
                    
                    sentiment_data.append({
                        'Sentiment': sentiment,
                        'Score': score
                    })
            except Exception as e:
                print(f"Skipping problematic review: {e}")
                continue
        
        if not sentiment_data:
            print("No valid sentiment data after processing")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(sentiment_data)
        
        # Map sentiments to standardized categories
        sentiment_map = {
            'positive': 'Positive',
            'pos': 'Positive',
            '+ve': 'Positive',
            'negative': 'Negative', 
            'neg': 'Negative',
            '-ve': 'Negative',
            'neutral': 'Neutral',
            'mixed': 'Mixed',
            'neutral (mixed)': 'Mixed',
            '0': 'Neutral'
        }
        
        # Apply sentiment mapping, with a default to 'Neutral'
        df['Sentiment'] = df['Sentiment'].apply(lambda x: sentiment_map.get(str(x).lower(), 'Neutral'))
        
        # Visualization
        plt.figure(figsize=(10, 6))
        
        # Define color palette
        colors = {
            'Positive': '#4CAF50',   # Green
            'Negative': '#F44336',   # Red
            'Neutral': '#2196F3',    # Blue
            'Mixed': '#FFC107'       # Amber
        }
        
        # Count sentiments and get default color for unknown categories
        sentiment_counts = df['Sentiment'].value_counts()
        
        # Ensure we have a color for each category
        plot_colors = [colors.get(s, '#9E9E9E') for s in sentiment_counts.index]
        
        # Create bar plot
        bars = plt.bar(
            sentiment_counts.index, 
            sentiment_counts.values, 
            color=plot_colors
        )
        
        plt.title('Review Sentiment Distribution', fontsize=15)
        plt.xlabel('Sentiment Category', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        
        # Add percentage annotations
        total = len(df)
        for i, (category, count) in enumerate(sentiment_counts.items()):
            percentage = count / total * 100
            plt.text(
                i, count, 
                f'{count} ({percentage:.1f}%)', 
                ha='center', va='bottom'
            )
        
        plt.tight_layout()
        return plt.gcf()
    
    except Exception as e:
        print(f"Sentiment distribution plot error: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_sentiment_trend(reviews, min_reviews=1):
    """
    Robust sentiment trend visualization
    
    Args:
        reviews (list): List of review dictionaries
        min_reviews (int): Minimum number of reviews required to generate plot
    
    Returns:
        matplotlib.figure.Figure or None
    """
    try:
        # Validate reviews
        if not reviews or len(reviews) < min_reviews:
            print(f"Not enough reviews for trend (need {min_reviews}, got {len(reviews) if isinstance(reviews, list) else 'not a list'})")
            return None
        
        # Prepare data with robust date parsing and score extraction
        trend_data = []
        for review in reviews:
            try:
                # Handle string reviews by skipping (no date information)
                if isinstance(review, str):
                    continue
                
                # Handle dictionary reviews
                if isinstance(review, dict):
                    # Try to get date with fallbacks
                    date_str = review.get('Review Date', 
                                review.get('Date', 
                                review.get('review_date', None)))
                    
                    # Skip if no date information
                    if not date_str:
                        continue
                    
                    # Try multiple date parsing approaches
                    try:
                        date = pd.to_datetime(date_str, errors='coerce')
                    except:
                        try:
                            date = dateparser.parse(date_str)
                        except:
                            continue
                    
                    if pd.isna(date):
                        continue
                    
                    # Get sentiment score with fallbacks
                    if 'Sentiment Score' in review:
                        sentiment_score = float(review.get('Sentiment Score', 0))
                    elif 'Review Content' in review and isinstance(review['Review Content'], str):
                        # Calculate sentiment if we have review content
                        sentiment_result = analyze_sentiment(review['Review Content'])
                        sentiment_score = sentiment_result['compound']
                    else:
                        # Default score
                        sentiment_score = 0
                    
                    trend_data.append({
                        'Date': date,
                        'Score': sentiment_score
                    })
            except Exception as e:
                print(f"Skipping problematic review for trend: {e}")
                continue
        
        if not trend_data or len(trend_data) < min_reviews:
            print(f"Not enough valid trend data after processing (need {min_reviews}, got {len(trend_data)})")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(trend_data)
        df.sort_values('Date', inplace=True)
        
        # Resample and smooth
        df.set_index('Date', inplace=True)
        
        # Handle case with too few data points for weekly resampling
        if len(df) < 7:
            # Just use the raw data points
            smoothed_trend = df
        else:
            # Resample to weekly frequency
            smoothed_trend = df.resample('W')['Score'].mean().fillna(method='ffill')
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed_trend.index, smoothed_trend.values, 
                 marker='o', linestyle='-', color='#2196F3')
        
        plt.title('Sentiment Trend Over Time', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        
        plt.axhline(y=0, color='gray', linestyle='--')  # Neutral line
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        print(f"Sentiment trend plot error: {e}")
        import traceback
        traceback.print_exc()
        return None
def main():
    st.title("E-commerce Product Scraper")
    st.write("Compare products from major e-commerce platforms")

    # Initialize session state
    if 'structured_data' not in st.session_state:
        st.session_state.structured_data = None
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'product_details' not in st.session_state:
        st.session_state.product_details = None
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0

    # Website selection
    website = st.selectbox("Select Website", ["Amazon", "Flipkart", "Myntra"])
    query = st.text_input("Enter product search query", "shoes")

    if st.button("Search Products"):
        with st.spinner("Searching products..."):
            search_url = generate_search_url(website, query)
            if search_url:
                crawl_data = crawl_website(search_url)
                if crawl_data:
                    st.session_state.structured_data = structure_data_with_gemini(crawl_data, website, query)
                    if st.session_state.structured_data:
                        st.success(f"Found {len(st.session_state.structured_data)} products!")
                    else:
                        st.error("No products found or error structuring data.")

    if st.session_state.structured_data:
        st.subheader("Search Results")
        
        # Display product cards
        cols = st.columns(3)
        for index, product in enumerate(st.session_state.structured_data):
            with cols[index % 3]:
                with st.container():
                    st.markdown("""
                        <style>
                            .card {
                                padding: 15px;
                                border-radius: 10px;
                                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                                margin-bottom: 20px;
                            }
                            .card img {
                                max-height: 200px;
                                object-fit: contain;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    with st.container():
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        image_url = product.get("Image URL")
                        if image_url:
                            st.image(image_url, use_column_width=True)
                        st.markdown(f"**{product.get('Product Name', 'N/A')}**")
                        st.markdown(f"**Price:** {product.get('Price', 'N/A')}")
                        rating = product.get('Rating')
                        if rating:
                            st.markdown(f"**Rating:** {rating}/5")
                        if st.button(f"Select #{index+1}", key=f"card_btn_{index}"):
                            st.session_state.selected_index = index
                        st.markdown("</div>", unsafe_allow_html=True)

        product_names = [f"{p.get('Product Name', 'N/A')} - {p.get('Price', 'N/A')}" 
                        for p in st.session_state.structured_data]
        selected_index = st.selectbox(
            "Or select a product from the list:",
            range(len(product_names)),
            index=st.session_state.selected_index,
            format_func=lambda x: product_names[x]
        )
        
        if selected_index != st.session_state.selected_index:
            st.session_state.selected_index = selected_index

        if st.button("View Product Details"):
            with st.spinner("Fetching product details..."):
                selected_product = st.session_state.structured_data[st.session_state.selected_index]
                product_url = selected_product.get("Product URL")
                if product_url:
                    st.session_state.product_details = crawl_product_details(product_url)
                    st.session_state.selected_product = selected_product
                    if not st.session_state.product_details:
                        st.error("Failed to fetch product details.")

    if st.session_state.product_details:
        st.subheader("Product Details")
        
        # Display product info
        with st.expander("Product Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** {st.session_state.product_details.get('Product Name', 'N/A')}")
                st.markdown(f"**Price:** {st.session_state.product_details.get('Price', 'N/A')}")
                st.markdown(f"**Brand:** {st.session_state.product_details.get('Brand', 'N/A')}")
            with col2:
                st.markdown(f"**Rating:** {st.session_state.product_details.get('Rating', 'N/A')}")
                st.markdown(f"**Reviews Count:** {st.session_state.product_details.get('Reviews', 'N/A')}")
        
        # MOVED OUTSIDE: Display description in separate expander
        if 'Description' in st.session_state.product_details and st.session_state.product_details['Description']:
            with st.expander("Product Description"):
                st.write(st.session_state.product_details['Description'])

        # Check if Review Content exists and is a list
        reviews = st.session_state.product_details.get('Review Content', [])
        
        # Debug information
        with st.expander("Review Data Structure"):
            st.write(f"Reviews Type: {type(reviews)}")
            st.write(f"Reviews Length: {len(reviews) if isinstance(reviews, (list, tuple)) else 'Not a list'}")
            if isinstance(reviews, (list, tuple)) and len(reviews) > 0:
                st.write(f"First Review Type: {type(reviews[0])}")
                if isinstance(reviews[0], dict):
                    st.write("First Review Keys:", list(reviews[0].keys()))
                elif isinstance(reviews[0], str):
                    st.write("First Review (string):", reviews[0][:100] + "..." if len(reviews[0]) > 100 else reviews[0])
                else:
                    st.write(f"First Review (other type): {reviews[0]}")
            else:
                st.write("No reviews available or not in expected format")

        # Ensure reviews is a list
        if not isinstance(reviews, (list, tuple)):
            st.warning(f"Review content is not a list. Got {type(reviews)}")
            # Try to convert to list if it's a string
            if isinstance(reviews, str):
                reviews = [{"Review Content": reviews, "Sentiment": "Unknown", "Sentiment Score": 0}]
            else:
                reviews = []
        
        # Display reviews if available
        if reviews:
            st.subheader("Sentiment Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sentiment Distribution")
                fig = plot_sentiment_distribution(reviews)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Could not generate sentiment distribution")
            
            with col2:
                st.markdown("### Sentiment Trend Over Time")
                fig = plot_sentiment_trend(reviews)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Could not generate sentiment trend")
            
            st.markdown("### Aspect-Based Analysis")
            if st.button("Analyze Product Aspects"):
                with st.spinner("Analyzing product aspects..."):
                    aspect_analysis = perform_aspect_analysis(reviews)
                    if aspect_analysis:
                        st.write("#### Key Product Aspects")
                        for aspect, data in aspect_analysis.items():
                            sentiment = data.get('sentiment', 'neutral')
                            examples = data.get('examples', [])
                            example_text = ", ".join(examples[:2]) if examples else "No examples"
                            st.markdown(f"""
                            **{aspect.capitalize()}** ({sentiment})
                            - Examples: {example_text}
                            """)
                    else:
                        st.error("Failed to perform aspect analysis")
            
            # Display review table
            with st.expander("Review Details"):
                if isinstance(reviews[0], dict):
                    # Create a simplified DataFrame for display
                    review_display = []
                    for r in reviews:
                        if isinstance(r, dict):
                            review_display.append({
                                "Content": r.get("Review Content", "")[:100] + "..." if len(r.get("Review Content", "")) > 100 else r.get("Review Content", ""),
                                "Rating": r.get("Rating", ""),
                                "Sentiment": r.get("Sentiment", ""),
                                "Date": r.get("Review Date", "")
                            })
                    
                    if review_display:
                        st.dataframe(pd.DataFrame(review_display))
                    else:
                        st.write("No valid review data to display")
                else:
                    st.write("Review data is not in the expected dictionary format")
        else:
            st.warning("No reviews available for sentiment analysis")

        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Product Data to Excel"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"product_data_{website}_{timestamp}.xlsx"
                
                # Create a clean version of product details without review content
                export_data = dict(st.session_state.product_details)
                if "Review Content" in export_data:
                    export_data["Review Content"] = f"{len(reviews)} reviews available"
                
                df = pd.DataFrame([export_data])
                
                towrite = io.BytesIO()
                df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)

        with col2:
            if st.button("Export Reviews to Excel"):
                if reviews:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"reviews_{website}_{timestamp}.xlsx"
                    
                    # Create a clean data frame from reviews
                    review_data = []
                    for r in reviews:
                        if isinstance(r, dict):
                            review_data.append(r)
                        elif isinstance(r, str):
                            review_data.append({"Review Content": r})
                    
                    if review_data:
                        df = pd.DataFrame(review_data)
                        
                        towrite = io.BytesIO()
                        df.to_excel(towrite, index=False, engine='openpyxl')
                        towrite.seek(0)
                        b64 = base64.b64encode(towrite.read()).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Reviews</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("No valid review data to export")
                else:
                    st.warning("No reviews available for this product")

if __name__ == "__main__":
    main()
