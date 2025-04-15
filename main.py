import os
import json
import re
from firecrawl import FirecrawlApp
import requests
from dotenv import load_dotenv
import pandas as pd 
from datetime import datetime 

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

if not firecrawl_api_key:
    print("Error: FIRECRAWL_API_KEY is missing")
    exit()

if not deepseek_api_key:
    print("Error: DEEPSEEK_API_KEY is missing")
    exit()

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

def clean_price(price_str):
    """Convert price string to clean format"""
    if not price_str:
        return None
   
    price_str = price_str.replace('\\u20b9', '₹')
   
    numbers = re.findall(r'[\d,]+\.?\d*', price_str)
    if numbers:
       
        return '₹' + numbers[0].replace(',', '')
    return None

def crawl_all_reviews_pages(product_url, base_product_data=None):
    """Crawl all pages of reviews for a product"""
    print(f"Crawling all review pages for: {product_url}")
    
    all_reviews = []
    current_page_url = product_url
    page_num = 1
    
    # Some sites have review-specific URLs
    if "amazon" in product_url.lower():
        # Convert to reviews page if it's not already
        if "/dp/" in product_url and "/reviews/" not in product_url:
            product_id = re.search(r'/dp/([A-Z0-9]+)', product_url).group(1)
            current_page_url = f"https://www.amazon.in/product-reviews/{product_id}"
    
    while True:
        print(f"Crawling reviews page {page_num}...")
        page_data = app.scrape_url(current_page_url)
        
        if not page_data:
            print(f"Failed to get data for reviews page {page_num}")
            break
            
        # Process first page differently if we're also getting product details
        if page_num == 1 and base_product_data is None:
            structured_details = structure_data_with_deepseek(page_data)
            if structured_details and isinstance(structured_details, list):
                base_product_data = structured_details[0]
            elif structured_details:
                base_product_data = structured_details
                
        # Extract reviews from this page
        page_reviews = extract_reviews_details(page_data)
        
        if not page_reviews or len(page_reviews) == 0:
            print(f"No reviews found on page {page_num} or reached the end of reviews")
            break
            
        all_reviews.extend(page_reviews)
        print(f"Found {len(page_reviews)} reviews on page {page_num}. Total: {len(all_reviews)}")
        
        # Look for next page link
        next_page_url = find_next_reviews_page(page_data, current_page_url, page_num)
        if not next_page_url or next_page_url == current_page_url:
            print("No more review pages found")
            break
            
        current_page_url = next_page_url
        page_num += 1
        
    return all_reviews, base_product_data

def find_next_reviews_page(page_data, current_url, current_page_num):
    """Find the URL for the next page of reviews using DeepSeek"""
    try:
        print("Finding next review page with DeepSeek...")
        
        # DeepSeek API endpoint and headers
        DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = (
            "You are a web page analyzer. You're looking at a product reviews page. "
            "Find the URL for the next page of reviews from the HTML data. "
            "Only return the full next page URL - nothing else. "
            "If there is no next page, respond with just the word 'None'.\n\n"
            f"Current URL: {current_url}\n"
            f"Current page number: {current_page_num}\n\n"
            f"Page data: {json.dumps(page_data)}"
        )
        
        payload = {
            "model": "deepseek-coder",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts data from web content."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error: DeepSeek API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        response_data = response.json()
        next_page_url = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Handle the case where DeepSeek couldn't find a next page
        if next_page_url == "None" or not next_page_url.startswith("http"):
            # Fallback: try to construct the next page URL based on common patterns
            if "amazon" in current_url.lower():
                # Amazon pattern
                if "page=" in current_url:
                    return re.sub(r'page=\d+', f'page={current_page_num + 1}', current_url)
                elif "ref=" in current_url:
                    return current_url.split("ref=")[0] + f"ref=cm_cr_getr_d_paging_btm_next_{current_page_num + 1}"
                else:
                    return current_url + f"?pageNumber={current_page_num + 1}"
            elif "flipkart" in current_url.lower():
                # Flipkart pattern
                if "page=" in current_url:
                    return re.sub(r'page=\d+', f'page={current_page_num + 1}', current_url)
                else:
                    return current_url + f"&page={current_page_num + 1}"
            else:
                return None
                
        return next_page_url
    except Exception as e:
        print(f"Error finding next review page: {e}")
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
    text = text.replace("'", '"').replace("'", '"').replace("'", '"').replace("'", '"')
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

def structure_data_with_deepseek(crawl_data, website=None, search_query=None):
    """Structure scraped data using DeepSeek AI"""
    try:
        print("Structuring data with DeepSeek...")
        
        # DeepSeek API endpoint and headers
        DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        if not deepseek_api_key:
            print("Error: DEEPSEEK_API_KEY is missing.")
            return None
        
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare the prompt for DeepSeek
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
            "- Product URL (string)\n\n"
            "Format the response as a valid JSON array of objects. Include ONLY the JSON data, no explanations or markdown.\n\n"
            "Here's the data to process:\n"
            f"{json.dumps(crawl_data, indent=2)}"
        )
        
        # Prepare the payload for DeepSeek
        payload = {
            "model": "deepseek-coder",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from web content."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        # Make the API request to DeepSeek
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error: DeepSeek API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            safe_export_to_excel(crawl_data, website, search_query, is_structured=False)
            return None
        
        response_data = response.json()
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not response_text:
            print("Error: Received empty response from DeepSeek.")
            safe_export_to_excel(crawl_data, website, search_query, is_structured=False)
            return None
        
        # Clean and parse the response
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
            # Export the partial data
            safe_export_to_excel(response_text, website, search_query, is_structured=False)
            
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
        print(f"Error structuring data with DeepSeek: {e}")
        # Export the raw data in case of error
        safe_export_to_excel(crawl_data, website, search_query, is_structured=False)
        return None

def crawl_product_details(product_url):
    """Crawl the product details page and structure the data using DeepSeek"""
    try:
        print(f"Crawling product details page: {product_url}")
        product_data = app.scrape_url(product_url)
        print("Product details crawled successfully.")
        structured_details = structure_data_with_deepseek(product_data)
        if structured_details:
            print("Product details structured successfully.")
            # Extract reviews details
            reviews_data = extract_reviews_details(product_data)
            if reviews_data:
                structured_details[0]['Review Content'] = reviews_data
                # Add sentiment analysis
                sentiment_analysis = analyze_reviews_sentiment(reviews_data)
                if sentiment_analysis:
                    structured_details[0]['Sentiment Analysis'] = sentiment_analysis
                    export_sentiment_analysis(sentiment_analysis, structured_details[0], 
                                           website="", search_query="")
            return structured_details[0] if isinstance(structured_details, list) else structured_details
        else:
            print("Failed to structure product details.")
            return None
    except Exception as e:
        print(f"Error crawling product details: {e}")
        return None

def extract_reviews_details(crawl_data):
    """
    Extract detailed review information using DeepSeek
    """
    try:
        print("Extracting review details with DeepSeek...")
        
        DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
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
        
        payload = {
            "model": "deepseek-coder",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts review data from web content."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error: DeepSeek API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        response_data = response.json()
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not response_text or response_text == "None":
            print("Error: Received empty response from DeepSeek.")
            return None
            
        if response_text.startswith("```"):
            blocks = response_text.split("```")
            for block in blocks:
                if block.strip().startswith("[") or block.strip().startswith("{"):
                    response_text = block.strip()
                    break
        
        structured_reviews = json.loads(repair_json(response_text))
        return structured_reviews if isinstance(structured_reviews, list) else [structured_reviews]
        
    except Exception as e:
        print(f"Error extracting reviews: {e}")
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
                reviews_df.to_excel(writer, sheet_name='Reviews', index=False)
            
            # Metadata Sheet
            metadata = pd.DataFrame({
                'Information': ['Website', 'Search Query', 'Export Date', 'Number of Reviews'],
                'Value': [
                    website,
                    search_query,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    len(reviews_data) if reviews_data else 0
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
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

def safe_export_to_excel(data_to_export, website, search_query, is_structured=True):
    """
    Safely export any data to Excel, handling partial or malformed data
    
    Parameters:
    data_to_export: Can be dict, list, str, or any other data type
    website (str): Name of the website
    search_query (str): Search query used
    is_structured (bool): Whether the data is already structured
    
    Returns:
    str: Path to the saved Excel file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"product_data_{website}_{timestamp}.xlsx"
    
    try:
        # Handle different types of data
        if isinstance(data_to_export, dict):
            # For dictionary data
            df = pd.DataFrame(list(data_to_export.items()), columns=['Property', 'Value'])
        elif isinstance(data_to_export, list):
            # For list of dictionaries
            df = pd.DataFrame(data_to_export)
        elif isinstance(data_to_export, str):
            # For string data (like raw JSON or error messages)
            df = pd.DataFrame({'Raw Data': [data_to_export]})
        else:
            # For any other type of data
            df = pd.DataFrame({'Data': [str(data_to_export)]})
        
        # Create Excel writer object
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Write the main data
            df.to_excel(writer, sheet_name='Product Data', index=False)
            
            # Add metadata
            metadata = pd.DataFrame({
                'Information': ['Website', 'Search Query', 'Export Date', 'Data Status'],
                'Value': [
                    website,
                    search_query,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Structured' if is_structured else 'Raw/Partial'
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
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
        
        print(f"\nData exported successfully to {filename}")
        return filename
    except Exception as e:
        print(f"Error during Excel export: {str(e)}")
        # Last resort: try to save raw data
        try:
            with open(f"raw_data_{timestamp}.txt", 'w') as f:
                f.write(str(data_to_export))
            print(f"Raw data saved to raw_data_{timestamp}.txt")
        except:
            print("Failed to save raw data")
        return None

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

def analyze_sentiment_with_deepseek(review_text):
    """Analyze sentiment of review text using DeepSeek"""
    try:
        DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = (
            "Analyze the sentiment and key aspects of this product review. Return a JSON object with:\n"
            "- sentiment_score: number between -1 (very negative) to 1 (very positive)\n"
            "- sentiment_label: (Positive/Negative/Neutral)\n"
            "- key_pros: array of main positive points\n"
            "- key_cons: array of main negative points\n"
            "- emotional_tone: dominant emotion (e.g., satisfied, frustrated, delighted, disappointed)\n"
            "- purchase_recommendation: boolean indicating if reviewer seems to recommend the product\n"
            "- helpful_for_vendor: key actionable insights for the vendor\n\n"
            f"Review: {review_text}"
        )
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a sentiment analysis assistant that processes product reviews."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error: DeepSeek API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        response_data = response.json()
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        return json.loads(repair_json(response_text))
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None

def analyze_reviews_sentiment(reviews_data):
    """Analyze sentiment for all reviews and generate insights"""
    if not reviews_data:
        return None
        
    sentiment_results = []
    overall_stats = {
        "total_reviews": len(reviews_data),
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "common_pros": {},
        "common_cons": {},
        "emotional_distribution": {},
        "recommendation_rate": 0
    }

    for review in reviews_data:
        review_content = review.get("Review Content", "")
        if not review_content:
            continue
            
        sentiment_data = analyze_sentiment_with_deepseek(review_content)
        if not sentiment_data:
            continue
            
        sentiment_results.append({
            "review_id": reviews_data.index(review),
            "sentiment_data": sentiment_data
        })
        
        # Update overall statistics
        sentiment_label = sentiment_data.get("sentiment_label", "")
        if sentiment_label.lower() == "positive":
            overall_stats["positive_count"] += 1
        elif sentiment_label.lower() == "negative":
            overall_stats["negative_count"] += 1
        else:
            overall_stats["neutral_count"] += 1
            
        # Update common pros
        for pro in sentiment_data.get("key_pros", []):
            pro_key = pro.lower()
            overall_stats["common_pros"][pro_key] = overall_stats["common_pros"].get(pro_key, 0) + 1
            
        # Update common cons
        for con in sentiment_data.get("key_cons", []):
            con_key = con.lower()
            overall_stats["common_cons"][con_key] = overall_stats["common_cons"].get(con_key, 0) + 1
            
        # Update emotional distribution
        emotion = sentiment_data.get("emotional_tone", "").lower()
        if emotion:
            overall_stats["emotional_distribution"][emotion] = overall_stats["emotional_distribution"].get(emotion, 0) + 1
            
        # Update recommendation rate
        if sentiment_data.get("purchase_recommendation", False):
            overall_stats["recommendation_rate"] += 1
    
    # Calculate percentages and sort common factors
    if overall_stats["total_reviews"] > 0:
        overall_stats["positive_percentage"] = (overall_stats["positive_count"] / overall_stats["total_reviews"]) * 100
        overall_stats["negative_percentage"] = (overall_stats["negative_count"] / overall_stats["total_reviews"]) * 100
        overall_stats["neutral_percentage"] = (overall_stats["neutral_count"] / overall_stats["total_reviews"]) * 100
        overall_stats["recommendation_percentage"] = (overall_stats["recommendation_rate"] / overall_stats["total_reviews"]) * 100
    
    # Sort common pros and cons by frequency
    overall_stats["common_pros"] = dict(sorted(overall_stats["common_pros"].items(), key=lambda x: x[1], reverse=True)[:10])
    overall_stats["common_cons"] = dict(sorted(overall_stats["common_cons"].items(), key=lambda x: x[1], reverse=True)[:10])
    overall_stats["emotional_distribution"] = dict(sorted(overall_stats["emotional_distribution"].items(), key=lambda x: x[1], reverse=True))
    
    return {
        "individual_reviews": sentiment_results,
        "overall_stats": overall_stats
    }

def export_sentiment_analysis(sentiment_data, product_info, website, search_query):
    """Export sentiment analysis results to Excel"""
    if not sentiment_data:
        print("No sentiment data to export.")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_analysis_{website}_{timestamp}.xlsx"
    
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
            
            # Overall Stats Sheet
            overall_stats = sentiment_data.get('overall_stats', {})
            if overall_stats:
                stats_df = pd.DataFrame({
                    'Metric': [
                        'Total Reviews', 
                        'Positive Reviews (%)',
                        'Negative Reviews (%)',
                        'Neutral Reviews (%)',
                        'Recommendation Rate (%)'
                    ],
                    'Value': [
                        overall_stats.get('total_reviews', 0),
                        round(overall_stats.get('positive_percentage', 0), 2),
                        round(overall_stats.get('negative_percentage', 0), 2),
                        round(overall_stats.get('neutral_percentage', 0), 2),
                        round(overall_stats.get('recommendation_percentage', 0), 2)
                    ]
                })
                stats_df.to_excel(writer, sheet_name='Overall Stats', index=False)
                
                # Common Pros Sheet
                pros = overall_stats.get('common_pros', {})
                if pros:
                    pros_df = pd.DataFrame({
                        'Pro': list(pros.keys()),
                        'Count': list(pros.values())
                    })
                    pros_df.to_excel(writer, sheet_name='Common Pros', index=False)
                
                # Common Cons Sheet
                cons = overall_stats.get('common_cons', {})
                if cons:
                    cons_df = pd.DataFrame({
                        'Con': list(cons.keys()),
                        'Count': list(cons.values())
                    })
                    cons_df.to_excel(writer, sheet_name='Common Cons', index=False)
                
                # Emotional Distribution Sheet
                emotions = overall_stats.get('emotional_distribution', {})
                if emotions:
                    emotions_df = pd.DataFrame({
                        'Emotion': list(emotions.keys()),
                        'Count': list(emotions.values())
                    })
                    emotions_df.to_excel(writer, sheet_name='Emotional Distribution', index=False)
            
            # Individual Reviews Sheet
            individual_reviews = sentiment_data.get('individual_reviews', [])
            if individual_reviews:
                reviews_data = []
                for review_item in individual_reviews:
                    review_id = review_item.get('review_id', 'N/A')
                    sentiment_item = review_item.get('sentiment_data', {})
                    reviews_data.append({
                        'Review ID': review_id,
                        'Sentiment Score': sentiment_item.get('sentiment_score', 'N/A'),
                        'Sentiment Label': sentiment_item.get('sentiment_label', 'N/A'),
                        'Emotional Tone': sentiment_item.get('emotional_tone', 'N/A'),
                        'Purchase Recommendation': sentiment_item.get('purchase_recommendation', 'N/A'),
                        'Key Pros': ', '.join(sentiment_item.get('key_pros', [])),
                        'Key Cons': ', '.join(sentiment_item.get('key_cons', [])),
                        'Vendor Insights': sentiment_item.get('helpful_for_vendor', 'N/A')
                    })
                reviews_df = pd.DataFrame(reviews_data)
                reviews_df.to_excel(writer, sheet_name='Individual Reviews', index=False)
            
            # Metadata Sheet
            metadata = pd.DataFrame({
                'Information': ['Website', 'Search Query', 'Export Date', 'Number of Reviews Analyzed'],
                'Value': [
                    website,
                    search_query,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    overall_stats.get('total_reviews', 0)
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
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
        
        print(f"\nSentiment analysis exported successfully to {filename}")
        return filename
    except Exception as e:
        print(f"Error during sentiment analysis export: {str(e)}")
        return None

def main():
    """Main function to run the crawler with user input"""
    print("=" * 80)
    print("E-Commerce Product Crawler")
    print("=" * 80)
    
    while True:
        print("\nSelect an option:")
        print("1. Search for products")
        print("2. Crawl specific product URL")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Search for products
            website = input("\nEnter website (Amazon/Flipkart/Myntra): ").strip()
            search_query = input("Enter search query: ").strip()
            
            if not website or not search_query:
                print("Error: Website and search query are required.")
                continue
                
            search_url = generate_search_url(website, search_query)
            if not search_url:
                continue
                
            crawl_data = crawl_website(search_url)
            if not crawl_data:
                print("Failed to crawl the website. Please try again.")
                continue
                
            structured_data = structure_data_with_deepseek(crawl_data, website, search_query)
            if not structured_data:
                print("Failed to structure the data. Trying manual parsing...")
                structured_data = structure_data_manually(crawl_data)
                
            if structured_data:
                print(f"\nFound {len(structured_data)} products.")
                
                # Export to Excel
                safe_export_to_excel(structured_data, website, search_query)
                
                # Display limited results in console
                for idx, product in enumerate(structured_data[:5]):
                    print(f"\nProduct {idx+1}:")
                    cleaned_product = validate_product(product)
                    for key, value in cleaned_product.items():
                        if key != "Review Content":
                            print(f"- {key}: {value}")
                    
                    # Ask if user wants to crawl product details
                    if idx == 0:  # Just for the first product
                        crawl_details = input("\nDo you want to crawl detailed reviews for this product? (y/n): ").strip().lower()
                        if crawl_details == 'y':
                            product_url = cleaned_product.get('Product URL')
                            if product_url:
                                reviews, product_data = crawl_all_reviews_pages(product_url, cleaned_product)
                                if reviews:
                                    print(f"Successfully crawled {len(reviews)} reviews.")
                                    export_reviews_to_excel(reviews, product_data, website, search_query)
                                    
                                    # Analyze sentiment
                                    sentiment_analysis = analyze_reviews_sentiment(reviews)
                                    if sentiment_analysis:
                                        export_sentiment_analysis(sentiment_analysis, product_data, website, search_query)
                                else:
                                    print("No reviews found for this product.")
            else:
                print("Failed to extract structured data.")
                
        elif choice == "2":
            # Crawl specific product URL
            product_url = input("\nEnter product URL: ").strip()
            
            if not product_url:
                print("Error: Product URL is required.")
                continue
                
            # Extract website from URL
            website = "Unknown"
            if "amazon" in product_url.lower():
                website = "Amazon"
            elif "flipkart" in product_url.lower():
                website = "Flipkart"
            elif "myntra" in product_url.lower():
                website = "Myntra"
                
            # Crawl all review pages
            reviews, product_data = crawl_all_reviews_pages(product_url)
            
            if product_data and reviews:
                print(f"Successfully crawled product details and {len(reviews)} reviews.")
                display_product_details(product_data)
                export_reviews_to_excel(reviews, product_data, website, "direct_url")
                
                # Analyze sentiment
                sentiment_analysis = analyze_reviews_sentiment(reviews)
                if sentiment_analysis:
                    export_sentiment_analysis(sentiment_analysis, product_data, website, "direct_url")
            elif product_data:
                print("Successfully crawled product details but no reviews found.")
                display_product_details(product_data)
                safe_export_to_excel(product_data, website, "direct_url")
            else:
                print("Failed to crawl product details. Trying direct approach...")
                product_details = crawl_product_details(product_url)
                
                if product_details:
                    print("Successfully crawled product details.")
                    display_product_details(product_details)
                    safe_export_to_excel(product_details, website, "direct_url")
                else:
                    print("Failed to extract product details.")
                    
        elif choice == "3":
            print("\nExiting the application. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            
if __name__ == "__main__":
    main() 