import os
import json
from argparse import ArgumentParser
import generate_parameters as parameters
from random import randint
import requests 

from pydantic import BaseModel

class AnswerFormat(BaseModel):
    explanations: list
    reasons: list
    references: list
    text_summary: str

def random_stocks(n):
    stocks = parameters.fetch_sp500_stocks()
    return [stocks[randint(0, len(stocks) - 1)] for _ in range(n)]

def fetch_historical_data(ticker):
    return parameters.fetch_historical_data(ticker)

def api_data_request(api_key, stock, date, output_file):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    setting = "You are a stock wizard, generating explanations for a drop in stock performance for a given range of time. You will be evaluated on the quality and novelty of your explanations in that order. Quality is measured by the degree to which your data is supported by the references you present. Novelty is the likelihood that another model would not present this information. Be concise and to the point. If any of the explanations appear weak, ignore them and focus on improving the others."
    query = f"Analyze the stock {stock} and provide detailed potential explanations for the drop in stock performance between the dates {date[0]} and {date[1]}. To accomplish this task, search for financial news for the given time period. Focus on the most relevant information. AVOID REPEATING YOURSELF. Look for and record any unusual patterns or events that may have caused the drop. Provide a summary of your findings, as well as any references you used to make your conclusions. Be sure to include the reasons for the drop in stock performance."
    
    response = requests.post(url, json={
        "model": "sonar",
        "messages": [{"role": "system", "content": setting}, {"role": "user", "content": query}],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 2,
        "response_format": {"type": "json_schema", "json_schema": {"schema": AnswerFormat.model_json_schema()}},
    }, headers=headers)
    
    if response.status_code == 200:
        full_data = response.json()
        filtered_data = {
            "citations": full_data.get("citations", []),
            "content": [choice.get("content") for choice in full_data.get("choices", [])]
        }
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f)
    else:
        print(f"Error {response.status_code}: {response.text}")

def generate_data(api_key, nrow=10**2, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stocks = random_stocks(nrow)
    for stock in stocks:
        historical_data = fetch_historical_data(stock[0])
        strange_ranges = parameters.fetch_strange_dates(historical_data)
        for dates in strange_ranges:
            output_file = os.path.join(output_dir, f"{stock[0]}_{dates[0]}_{dates[1]}.json")
            api_data_request(api_key, stock, dates, output_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="API key for authentication")
    parser.add_argument("--nrow", type=int, default=10**6, help="Number of random stocks to generate data for")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated data")
    args = parser.parse_args()
    generate_data(args.api_key, args.nrow, args.output_dir)

# python synthetic_data/generate_data.py --api_key ENTER_KEY_HERE --nrow 1000 --output_dir synthetic_data/json_files