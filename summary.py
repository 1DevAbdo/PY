import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from urllib.parse import urlparse


def scrape_articles():
    url = "https://news.sky.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    articles = soup.find_all("a", class_="sdc-site-tile__headline-link")

    # Limit the number of articles to scrape
    articles = articles[:5]

    news_data = []

    model = BartForConditionalGeneration.from_pretrained("Yelyzaveta/facebook-bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("Yelyzaveta/facebook-bart-large-cnn")

    for article in articles:
        title = article.text.strip()
        link = article["href"]

        if not link.startswith("http"):
            base_url = urlparse(url)
            link = base_url.scheme + "://" + base_url.netloc + link

        article_response = requests.get(link)
        article_soup = BeautifulSoup(article_response.content, "html.parser")

        main_article = article_soup.find("div", class_="sdc-article-body")
        if main_article:
            paragraphs = main_article.find_all("p")
            article_text = " ".join([p.text.strip() for p in paragraphs])
        else:
            article_text = ""

        summary = article_text[:512] + "..." if len(article_text) > 512 else article_text

        key_points_container = article_soup.find("div", {"data-test-id": "key-points"})
        if key_points_container:
            key_points = key_points_container.find_all("span", class_="sdc-article-key-point")
            formatted_key_points = "\n".join([point.text.strip() for point in key_points])
        else:
            formatted_key_points = ""

        keywords = []
        keywords_link = article_soup.find("a", string="View post")
        if keywords_link:
            keywords_href = keywords_link["href"]
            keywords.append(keywords_href)

        inputs = tokenizer.encode(summary, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, num_beams=4, max_length=200, early_stopping=True)
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        news_data.append({
            "title": title,
            "summary": summary,
            "link": link,
            "key_points": formatted_key_points,
            "keywords": keywords,
            "summarized_text": summarized_text
        })

    return news_data
