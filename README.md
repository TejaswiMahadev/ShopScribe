# 🛍️ E-commerce Product Scraper & Review Analyzer
This project is a powerful and intelligent Streamlit-based web application that scrapes, structures, and analyzes product listings and customer reviews from major Indian e-commerce platforms—Amazon, Flipkart, and Myntra. It uses Firecrawl for web scraping, Google Gemini for data structuring, and VADER for sentiment analysis.

## 🔍 Features
- **Product Search**: Search any product on Amazon, Flipkart, or Myntra.

- **Crawling & Scraping**: Fetch live data from product listing pages using Firecrawl API.

- **Structured Data Extraction**: Use Gemini (Google Generative AI) to structure unstructured data.

- **Review Sentiment Analysis**: Classify reviews as positive, negative, neutral, or mixed using VADER.

- **Aspect-Based Analysis**: Identify key product aspects and their sentiment.

- **Trend Visualization**: See how sentiment changes over time.


## 🧠 Technologies Used

- **Streamlit** – Web interface
  
- **Firecrawl API** – Website scraping engine
  
- **Google Gemini API  (via google.generativeai) ** – Generative AI for JSON structuring
  
- **VADER Sentiment** – NLP-based sentiment scoring
  
- **Pandas / Matplotlib / Seaborn** – Data wrangling & visualization

 ## ✨ Core Functionalities
 - **🔗 Website Crawling** : Dynamically generates search URLs for each platform. Scrapes product listings and individual product pages.

- **🧠 Review Extraction with AI** : Uses Gemini to extract and structure reviews from raw HTML content. Recovers partial data in case of parsing failures using robust fallback mechanisms.

- **❤️ Sentiment Analysis** : Scores each review using VADER (compound, positive, neutral, negative). Classifies into positive/neutral/negative/mixed.

- **📊 Visualizations** :
  - Bar Chart: Distribution of sentiments.
  - Line Graph: Trend of sentiment scores over time.

- **Aspect Analysis**: Uses Gemini to detect product-specific themes like "comfort", "battery life", etc.

