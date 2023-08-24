import requests
from textblob import TextBlob

# Replace YOUR_API_KEY with your actual API key
api_key = "apikey"

#ticker
word = "CHANDRAYAAN"

# Send a request to the News API to get articles about the word
url = f"https://newsapi.org/v2/everything?q={word}&apiKey={api_key}"
response = requests.get(url)

# Check the status code to make sure the request was successful
if response.status_code != 200:
    print('Failed to fetch data')
    exit()

# Get the data as a JSON object
data = response.json()

# Print the number of articles returned
print(f'Number of articles: {data["totalResults"]}')

# Iterate through the articles and print the title and publication date
for article in data['articles']:
    print(f'{article["title"]} ({article["publishedAt"]})')

# Initialize counters for positive, negative, and neutral articles
positive_count = 0
negative_count = 0
neutral_count = 0

for article in data['articles']:
    # Use TextBlob to perform sentiment analysis on the article
    blob = TextBlob(article['title'])
    sentiment = blob.sentiment.polarity

    # Classify the article as positive, negative, or neutral based on the sentiment
    if sentiment > 0:
        positive_count += 1
    elif sentiment < 0:
        negative_count += 1
    else:
        neutral_count += 1

# Print the results
print(f'Positive articles: {positive_count} ({positive_count / len(data["articles"]) * 100:.2f}%)')
print(f'Negative articles: {negative_count} ({negative_count / len(data["articles"]) * 100:.2f}%)')
print(f'Neutral articles: {neutral_count} ({neutral_count / len(data["articles"]) * 100:.2f}%)')
