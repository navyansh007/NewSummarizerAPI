from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from groq import Groq
from exa_py import Exa
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)
exa = Exa(
    api_key=os.environ.get("EXA_API_KEY")
)

@app.route('/')
@cross_origin()
def start():
    return render_template('index.html')


def getsearchquery(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates search queries based on user questions. Only generate one search query. Don't write anything else apart from the search query"
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ],
        model="llama3-70b-8192",
    )

    return chat_completion.choices[0].message.content.strip('"')

@app.route('/fetchurls', methods=['POST'])
@cross_origin()
def fetchurls():
    search_query = getsearchquery(request.json['query'])
    search_response = exa.search_and_contents(
        search_query, use_autoprompt=True, start_published_date=request.json['date_cutoff'], exclude_domains=['x.com', 'twitter.com'], category='news'
    )
    response = [{"url": result.url, "headline": result.title} for result in search_response.results]
    return jsonify({
        "result": response
    })

@app.route('/getsummary', methods=['POST'])
@cross_origin()
def getsummary():
    url = request.json['url']
    result = exa.get_contents(
        [url],
        text=True
    )
    result_text = result.results[0].text

    SYSTEM_MESSAGE = "You are a helpful assistant that briefly summarizes the given news article. Summarize the users input in not more than 100 words. Also, just directly start responding with the summarized text only."

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": result_text},
        ],
    )

    summary = completion.choices[0].message.content
    return jsonify({
        "url": url,
        "title": result.results[0].title,
        "author": result.results[0].author,
        "summary": summary
    })