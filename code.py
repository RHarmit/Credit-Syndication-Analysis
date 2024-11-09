import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Download NLTK's VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Your Alpha Vantage API key
api_key = '8K3FXFA00XIBZA9C'

# Step 1: Fetch 10-Year U.S. Treasury Yield Data (Using Alpha Vantage)
url = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={api_key}'
response = requests.get(url)
data = response.json()

# Check if data retrieval was successful
if 'data' in data:
    # Convert data to DataFrame and filter for the last 3 years
    yield_data = pd.DataFrame(data['data'])
    yield_data['date'] = pd.to_datetime(yield_data['date'])
    yield_data['value'] = yield_data['value'].astype(float)
    yield_data = yield_data[yield_data['date'] >= '2021-01-01']  # Filter for last 3 years
    
    # Sort data by date
    yield_data = yield_data.sort_values(by='date')
    
    # Plot the 10-year Treasury yield data over time
    plt.figure(figsize=(10, 6))
    plt.plot(yield_data['date'], yield_data['value'], label="10-Year Treasury Yield", color="blue")
    plt.title("10-Year U.S. Treasury Yield Over Last 3 Years")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Error fetching data:", data.get("Note", "Unknown error. Please check your API call or API limits."))

# Step 2: Fetch Real News Data from Alpha Vantage and Perform Sentiment Analysis
# Note: Replace 'function=NEWS_SENTIMENT' with the correct endpoint if it's updated in Alpha Vantage's documentation.
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=credit&apikey={api_key}'
news_response = requests.get(news_url)
news_data = news_response.json()

# Check if news data retrieval was successful
if 'feed' in news_data:
    # Extract headlines from the news feed
    headlines = [article['title'] for article in news_data['feed']]
    
    # Perform sentiment analysis on each headline
    sentiment_scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    print("\nNews Headlines and Sentiment Scores:")
    for headline, score in zip(headlines, sentiment_scores):
        print(f"Headline: {headline} | Sentiment Score: {score}")
    print("\nAverage Sentiment Score for Credit Market News:", average_sentiment)
else:
    print("Error fetching news data:", news_data.get("Note", "Unknown error. Please check your API call or API limits."))

# Step 3: 3-Year Portfolio Exposure and Deal Scoring
# Create sample data for deals and their respective spreads
syndication_data_3yr = {
    'Deal Name': ["Deal A", "Deal B", "Deal C"],
    'Industry': ["Tech", "Healthcare", "Finance"],
    'Credit Spread (%)': [3.5, 4.0, 4.5],  # Example credit spreads
    'Sentiment Score': [average_sentiment, average_sentiment - 0.2, average_sentiment + 0.1]  # Adjusted sentiment scores
}
portfolio_df_3yr = pd.DataFrame(syndication_data_3yr)

# Plot Credit Spread vs Sentiment Score for 3-year focused deals
plt.figure(figsize=(8, 6))
plt.scatter(portfolio_df_3yr['Credit Spread (%)'], portfolio_df_3yr['Sentiment Score'], color='blue')
plt.title('3-Year Credit Spread vs. Market Sentiment for Syndication Deals')
plt.xlabel('Credit Spread (%)')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()

# Step 4: Calculate 3-Year Deal Scores
# Calculate the score by multiplying credit spread and sentiment
portfolio_df_3yr['Deal Score'] = (portfolio_df_3yr['Credit Spread (%)'] * portfolio_df_3yr['Sentiment Score']).round(2)
print("\n3-Year Deal Scoring:\n", portfolio_df_3yr[['Deal Name', 'Deal Score']])

# Save results to a CSV for reference
portfolio_df_3yr.to_csv("3_year_syndication_deals.csv", index=False)
print("\nResults saved to '3_year_syndication_deals.csv'")
