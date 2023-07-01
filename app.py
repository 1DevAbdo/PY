from flask import Flask, render_template
from summary import scrape_articles

app = Flask(__name__, template_folder='templates', static_folder='static')

# Route for the home page
@app.route('/')
def home():
    # Scrape articles and store the results
    news_articles = scrape_articles()

    # Pass the data to the template for rendering
    return render_template('try.html', articles=news_articles)

if __name__ == '__main__':
    app.run(debug=True)
