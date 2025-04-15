import os
import json
import re
from firecrawl import FirecrawlApp
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd 
from datetime import datetime 
from tqdm import tqdm
import time 
import random 

#==============================CONFIGURATION=================================================
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

if not firecrawl_api_key:
    print("Error: FIRECRAWL_API_KEY is missing")
    exit()


app = FirecrawlApp(api_key=firecrawl_api_key)

#====================================CRAWLING LOGIC =========================================================

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

def crawl_website(url, max_retries=3, backoff_factor=2):
    """Crawl the specified URL using Firecrawl with retries"""
    for attempt in range(max_retries):
        try:
            print(f"Crawling website: {url} (attempt {attempt+1}/{max_retries})")
            crawl_data = app.scrape_url(url)
            print("Crawl successful. Data obtained.")
            return crawl_data
        except Exception as e:
            wait_time = backoff_factor ** attempt
            print(f"Error crawling website: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print(f"Failed to crawl {url} after {max_retries} attempts")
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

def improve_json_extraction(crawl_data):
    """Extract structured data from raw HTML using more intelligent parsing"""
    if isinstance(crawl_data, dict) and 'html' in crawl_data:
         
        try:
            import lxml.html
            from lxml.cssselect import CSSSelector
            
            html = lxml.html.fromstring(crawl_data['html'])
            
            # Look for JSON-LD
            script_tags = html.cssselect('script[type="application/ld+json"]')
            for script in script_tags:
                try:
                    json_data = json.loads(script.text_content())
                    if isinstance(json_data, dict) and '@type' in json_data:
                        if json_data['@type'] in ['Product', 'ProductPage']:
                            return json_data
                except:
                    continue
            selectors = {
                'amazon': {
                    'title': '#productTitle',
                    'price': '.a-price .a-offscreen',
                    'rating': '#acrPopover .a-icon-alt',
                    'reviews_count': '#acrCustomerReviewText',
                },
                'flipkart': {
                    'title': '.B_NuCI',
                    'price': '._30jeq3',
                    'rating': '._3LWZlK',
                    'reviews_count': '._2_R_DZ span',
                },
                
            }
            url = crawl_data.get('url', '')
            for site, site_selectors in selectors.items():
                if site in url:
                    result = {}
                    for key, selector in site_selectors.items():
                        elements = html.cssselect(selector)
                        if elements:
                            result[key] = elements[0].text_content().strip()
                    if result:
                        return result
        except Exception as e:
            print(f"Error extracting structured data: {e}")
            
    return crawl_data

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

def crawl_product_details(product_url):
    """Crawl the product details page and structure the data using Gemini"""
    try:
        print(f"Crawling product details page: {product_url}")
        product_data = app.scrape_url(product_url)
        print("Product details crawled successfully.")
        structured_details = structure_data_with_gemini(product_data)
        if structured_details:
            print("Product details structured successfully.")
            reviews_data = extract_reviews_details(product_data)
            if reviews_data:
                structured_details[0]['Review Content'] = reviews_data
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


def batch_process_products(website, query, num_products=5):
    """Process multiple products from a search query"""
    search_url = generate_search_url(website, query)
    results = []
    
    if search_url:
        crawl_data = crawl_website(search_url)
        
        if crawl_data:
            safe_export_to_excel(crawl_data, website, query, is_structured=False)
            
            structured_data = structure_data_with_gemini(crawl_data, website, query)
            if structured_data:
                safe_export_to_excel(structured_data, website, query, is_structured=True)
                products_to_process = structured_data[:min(num_products, len(structured_data))]
                
                print(f"\nProcessing {len(products_to_process)} products:")
                for i, product in enumerate(products_to_process, 1):
                    print(f"\n{i}. {product.get('Product Name', 'Unknown')} - {product.get('Price', 'Unknown')}")
                    
                    product_url = product.get("Product URL")
                    if product_url:
                        product_details = crawl_product_details(product_url)
                        if product_details:
                            all_reviews, _ = crawl_all_reviews_pages(product_url, product_details)
                            if all_reviews:
                              
                                product_details['Review Content'] = all_reviews
                                sentiment_analysis = analyze_reviews_sentiment(all_reviews)
                                if sentiment_analysis:
                                    product_details['Sentiment Analysis'] = sentiment_analysis
                                    export_sentiment_analysis(sentiment_analysis, product_details, 
                                                           website=website, search_query=query)
                                export_reviews_to_excel(all_reviews, product_details, website, query)
                            display_product_details(product_details)
                            results.append(product_details)
                safe_export_to_excel(results, website, query, is_structured=True)
                return results
            else:
                print(f"Error: Could not structure data from {website} for query '{query}'")
                return None
        else:
            print(f"Error: Could not crawl {website} for query '{query}'")
            return None
    else:
        print(f"Error: Could not generate search URL for {website}")
        return None

def get_random_user_agent():
    """Return a random user agent to avoid detection"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    return random.choice(user_agents)



#=========================================== CRAWLING REVIEWS=========================================

def crawl_all_reviews_pages(product_url, base_product_data=None):
    """Crawl all pages of reviews for a product"""
    print(f"Crawling all review pages for: {product_url}")
    
    all_reviews = []
    current_page_url = product_url
    page_num = 1
   
    if "amazon" in product_url.lower():
       
        if "/dp/" in product_url and "/reviews/" not in product_url:
            product_id = re.search(r'/dp/([A-Z0-9]+)', product_url).group(1)
            current_page_url = f"https://www.amazon.in/product-reviews/{product_id}"
    
    while True:
        print(f"Crawling reviews page {page_num}...")
        page_data = app.scrape_url(current_page_url)
        
        if not page_data:
            print(f"Failed to get data for reviews page {page_num}")
            break
            
       
        if page_num == 1 and base_product_data is None:
            structured_details = structure_data_with_gemini(page_data)
            if structured_details and isinstance(structured_details, list):
                base_product_data = structured_details[0]
            elif structured_details:
                base_product_data = structured_details
                
       
        page_reviews = extract_reviews_details(page_data)
        
        if not page_reviews or len(page_reviews) == 0:
            print(f"No reviews found on page {page_num} or reached the end of reviews")
            break
            
        all_reviews.extend(page_reviews)
        print(f"Found {len(page_reviews)} reviews on page {page_num}. Total: {len(all_reviews)}")
        
        next_page_url = find_next_reviews_page(page_data, current_page_url, page_num)
        if not next_page_url or next_page_url == current_page_url:
            print("No more review pages found")
            break
            
        current_page_url = next_page_url
        page_num += 1
        
    return all_reviews, base_product_data

def find_next_reviews_page(page_data, current_url, current_page_num):
    """Find the URL for the next page of reviews"""
    try:
       
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = (
            "You are a web page analyzer. You're looking at a product reviews page. "
            "Find the URL for the next page of reviews from the HTML data. "
            "Only return the full next page URL - nothing else. "
            "If there is no next page, respond with just the word 'None'.\n\n"
            f"Current URL: {current_url}\n"
            f"Current page number: {current_page_num}\n\n"
            f"Page data: {json.dumps(page_data)}"
        )
        response = model.generate_content(prompt)
        next_page_url = response.text.strip()
        
       
        if next_page_url == "None" or not next_page_url.startswith("http"):
            
            if "amazon" in current_url.lower():
               
                if "page=" in current_url:
                    return re.sub(r'page=\d+', f'page={current_page_num + 1}', current_url)
                elif "ref=" in current_url:
                    return current_url.split("ref=")[0] + f"ref=cm_cr_getr_d_paging_btm_next_{current_page_num + 1}"
                else:
                    return current_url + f"?pageNumber={current_page_num + 1}"
            elif "flipkart" in current_url.lower():
               
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


def structure_data_with_gemini(crawl_data, website=None, search_query=None):
    """Structure scraped data using Gemini AI with improved prompting"""
    try:
        print("Structuring data with Gemini...")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        enhanced_data = improve_json_extraction(crawl_data)
        
       
        example_format = {
            "amazon": {
                "Product Name": "ACME Wireless Headphones",
                "Price": "₹1,499",
                "Description": "Premium wireless headphones with noise cancellation",
                "Rating": "4.5 out of 5 stars",
                "Reviews": "2,345 ratings",
                "Brand": "ACME",
                "Product URL": "https://www.amazon.in/dp/B0123456789"
            },
            "flipkart": {
                "Product Name": "ACME Wireless Headphones",
                "Price": "₹1,499",
                "Description": "Premium wireless headphones with noise cancellation",
                "Rating": "4.5",
                "Reviews": "2,345 ratings",
                "Brand": "ACME",
                "Product URL": "https://www.flipkart.com/acme-wireless-headphones/p/itm123456789"
            }
        }
        
        website_example = example_format.get(website.lower(), example_format['amazon']) if website else example_format['amazon']
        
        prompt = (
            "You are a specialized e-commerce data extraction expert. "
            f"Extract product information from this scraped {website} search page for '{search_query}'. "
            "For each product on the page, provide these exact fields.\n\n"
            f"Example product format: {json.dumps(website_example, indent=2)}\n\n"
            "Follow these extraction rules:\n"
            "1. Return ONLY valid JSON with no explanations\n"
            "2. Extract ALL products visible on the page\n"
            "3. Include the complete product URL for each product\n"
            "4. If a field is not found, use null instead of empty string\n"
            "5. For prices, maintain currency symbols and formatting\n\n"
            "Here's the data to process:\n"
            f"{json.dumps(enhanced_data, indent=2)}"
        )
        
        response = model.generate_content(prompt)
        response = model.generate_content(prompt)
        
        if not response or not response.text.strip():
            print("Error: Received empty response from Generative AI.")
            safe_export_to_excel(crawl_data, website, search_query, is_structured=False)
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
            safe_export_to_excel(response_text, website, search_query, is_structured=False)
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
        safe_export_to_excel(crawl_data, website, search_query, is_structured=False)
        return None



def extract_reviews_details(crawl_data):
    """
    Extract detailed review information using Gemini AI
    """
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
        return structured_reviews if isinstance(structured_reviews, list) else [structured_reviews]
        
    except Exception as e:
        print(f"Error extracting reviews: {e}")
        return None

#============================================= EXPORT REVIEWS AND PRODUCT DETAILS===========================================================

def export_reviews_to_excel(reviews_data, product_info, website, search_query):
    """
    Export reviews data to a separate Excel file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"product_reviews_{website}_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
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
            if reviews_data:
                reviews_df = pd.DataFrame(reviews_data)
                reviews_df.to_excel(writer, sheet_name='Reviews', index=False)
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

# =========================================== SENTIMENT ANALYSIS===============================================================

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
        if isinstance(data_to_export, dict):
            df = pd.DataFrame(list(data_to_export.items()), columns=['Property', 'Value'])
        elif isinstance(data_to_export, list):
            df = pd.DataFrame(data_to_export)
        elif isinstance(data_to_export, str):
            df = pd.DataFrame({'Raw Data': [data_to_export]})
        else:
            df = pd.DataFrame({'Data': [str(data_to_export)]})
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Product Data', index=False)
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
        try:
            with open(f"raw_data_{timestamp}.txt", 'w') as f:
                f.write(str(data_to_export))
            print(f"Raw data saved to raw_data_{timestamp}.txt")
        except:
            print("Failed to save raw data")
        return None

def analyze_sentiment_with_gemini(review_text):
    """Analyze sentiment of review text using Gemini AI"""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
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
        response = model.generate_content(prompt)
        return json.loads(repair_json(response.text))
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

    print("Analyzing sentiment for all reviews...")
    for review in tqdm(reviews_data):
        review_content = review.get("Review Content", "")
        if not review_content:
            continue
            
        sentiment_analysis = analyze_sentiment_with_gemini(review_content)
        if sentiment_analysis:
            review["sentiment_analysis"] = sentiment_analysis
            sentiment_results.append(sentiment_analysis)
            sentiment = sentiment_analysis["sentiment_label"]
            if sentiment == "Positive":
                overall_stats["positive_count"] += 1
            elif sentiment == "Negative":
                overall_stats["negative_count"] += 1
            else:
                overall_stats["neutral_count"] += 1
            for pro in sentiment_analysis["key_pros"]:
                overall_stats["common_pros"][pro] = overall_stats["common_pros"].get(pro, 0) + 1
            for con in sentiment_analysis["key_cons"]:
                overall_stats["common_cons"][con] = overall_stats["common_cons"].get(con, 0) + 1
            emotion = sentiment_analysis["emotional_tone"]
            overall_stats["emotional_distribution"][emotion] = \
                overall_stats["emotional_distribution"].get(emotion, 0) + 1
                
            if sentiment_analysis["purchase_recommendation"]:
                overall_stats["recommendation_rate"] += 1
    if sentiment_results:
        overall_stats["recommendation_rate"] = (
            overall_stats["recommendation_rate"] / len(sentiment_results) * 100
        )
        overall_stats["common_pros"] = dict(
            sorted(overall_stats["common_pros"].items(), 
                  key=lambda x: x[1], reverse=True)[:5]
        )
        overall_stats["common_cons"] = dict(
            sorted(overall_stats["common_cons"].items(), 
                  key=lambda x: x[1], reverse=True)[:5]
        )
        
    return overall_stats

def export_sentiment_analysis(sentiment_data, product_info, website, search_query):
    """Export sentiment analysis results to Excel"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_analysis_{website}_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            stats_df = pd.DataFrame([{
                'Total Reviews': sentiment_data['total_reviews'],
                'Positive Reviews': sentiment_data['positive_count'],
                'Negative Reviews': sentiment_data['negative_count'],
                'Neutral Reviews': sentiment_data['neutral_count'],
                'Recommendation Rate': f"{sentiment_data['recommendation_rate']:.1f}%"
            }])
            stats_df.to_excel(writer, sheet_name='Overall Statistics', index=False)
            pros_cons_df = pd.DataFrame({
                'Top Pros': list(sentiment_data['common_pros'].keys()),
                'Frequency': list(sentiment_data['common_pros'].values()),
                'Top Cons': list(sentiment_data['common_cons'].keys()),
                'Frequency.1': list(sentiment_data['common_cons'].values())
            })
            pros_cons_df.to_excel(writer, sheet_name='Pros and Cons Analysis', index=False)
            emotions_df = pd.DataFrame({
                'Emotion': list(sentiment_data['emotional_distribution'].keys()),
                'Count': list(sentiment_data['emotional_distribution'].values())
            })
            emotions_df.to_excel(writer, sheet_name='Emotional Analysis', index=False)
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
        print(f"Error exporting sentiment analysis: {str(e)}")
        return None


def main():
    """Main function to run the script"""
    print("\n===== E-commerce Product and Review Crawler =====\n")
    website_options = list(BASE_URLS.keys())
    print("Supported websites:")
    for i, site in enumerate(website_options, 1):
        print(f"{i}. {site}")
    
    while True:
        try:
            website_choice = int(input("\nSelect website (enter number): "))
            if 1 <= website_choice <= len(website_options):
                website = website_options[website_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(website_options)}")
        except ValueError:
            print("Please enter a valid number")
    
    search_query = input("\nEnter search query: ").strip()
    
    while True:
        try:
            num_products = int(input("\nNumber of products to analyze (1-10): "))
            if 1 <= num_products <= 10:
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nStarting analysis for {num_products} {website} products matching '{search_query}'...")
    
    results = batch_process_products(website, search_query, num_products)
    
    if results:
        print(f"\nAnalysis complete! Processed {len(results)} products.")
        print("See the generated Excel files for detailed information.")
    else:
        print("\nAnalysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()