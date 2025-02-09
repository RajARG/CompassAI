import os
import json
import yfinance as yf
from argparse import ArgumentParser
import requests
from pydantic import BaseModel

def send_post_request(url, payload, headers):
    response = requests.post(url, json=payload, headers=headers)
    try:
        return response.json()
    except json.decoder.JSONDecodeError:
        return {"error": "Invalid JSON", "status_code": response.status_code, "response_text": response.text}

class AnswerFormat(BaseModel):
    explanations: list
    reasons: list
    references: list
    text_summary: str

def api_data_request(api_key, stock, start, end):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    setting = """You are a stock wizard, generating explanations for a drop in stock performance for a given range of time. You will be evaluated on the quality and novelty of your explanations in that order. Quality is measured by the degree to which your data is supported by the references you present. Novelty is the likelihood that another model would not present this information. Be concise and to the point. If any of the explanations appear weak, ignore them and focus on improving the others. DO NOT MENTIONED UNSUPPORTED HYPOTHETICAL CLAIMS. """
    
    query = f"""Analyze the stock {stock} and provide detailed potential explanations for the drop in stock performance between the dates {start} and {end}. To accomplish this task, search for financial news for the given time period. Focus on the most relevant information. AVOID REPEATING YOURSELF. Look for and record any unusual patterns or events that may have caused the drop. Provide a summary of your findings, as well as any references you used to make your conclusions. Be sure to include the reasons for the drop in stock performance. DO NOT MENTION MARKET VOLATILITY."""
    
    payload = {
        "model": "sonar",
        "messages": [{"role": "system", "content": setting}, {"role": "user", "content": query}],
        "max_tokens": 200,
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 2,
    }
    full_data = send_post_request(url, payload, headers)
    # Return citations and content if available
    if full_data.get('choices'):
        return {
            "citations": full_data.get("citations", []),
            "content": [choice.get("message", {}).get("content") for choice in full_data.get("choices", [])]
        }
    return {"citations": [], "content": []}

def api_enhancement_request(api_key, stock, start, end, explanations, references):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    setting =  """ You will be provided with a list of citations to various news sources and a text string describing possible explanations for a drop in stock performance. Your job is to analyze this string and then rethink the provided explanations. Check each of the sites and enhance the explanations by providing additional details and context. Feel free to remove explanations that don't make much sense. YOU WILL BE EXPECTED TO FIND MORE RESOURCES. You will be evaluated on the quality and novelty of your enhancements in that order. Quality is measured by the degree to which your data is supported by the references you present. Novelty is the likelihood that another model would not present this information. Be concise and to the point. If any of the explanations appear weak, ignore them and focus on improving the others. DO NOT MENTIONED UNSUPPORTED HYPOTHETICAL CLAIMS. YOUR OUTPUT MUST BE IN THE JSON FORMAT: {\"explanations\": [\"explanation1\", \"explanation2\"], \"reasons\": [\"reason1\", \"reason2\"], \"references\": [\"reference1\", \"reference2\"], \"text_summary\": \"summary\"}" """
    
    query = f"""I have these explanations for the drop in stock performance of {stock} between the dates {start} and {end}.
        I need you to enhance them by providing additional details and context. Check the provided citations for more information.
        Be sure to remove any explanations that don't make much sense. Provide a summary of your enhancements, as well as any references you used to make your conclusions. Be sure to include the reasons for the drop in stock performance. DO NOT MENTION MARKET VOLATILITY. YOUR RESPONSE MUST BE IN VALID JSON FORMAT. AND DO NOT COMMENT ON THE PREVIOUS STRING, JUST THE EXPLANATIONS. \\n explanations: {explanations}, \\n references: {references}"""
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": setting}, {"role": "user", "content": query}],
        "response_format": {"type": 'json_object'},
    }
    return send_post_request(url, payload, headers)

def generate_data(api_key_1, api_key_2, stock, start, end):
    simple_explanations = api_data_request(api_key_1, stock, start, end)
    complex_explanations = api_enhancement_request(api_key_2, stock, start, end, simple_explanations['content'], simple_explanations["citations"])
    # Check for choices key in the returned complex explanations
    choices = complex_explanations.get('choices')
    if choices and isinstance(choices, list) and choices[0].get('message'):
        return choices[0]['message'].get('content', 'No content available')
    return "No valid complex explanation returned."

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--api_key_1", type=str, required=True, help="API key for FDP")
    parser.add_argument("--api_key_2", type=str, required=True, help="API key for LLM")
    parser.add_argument("--stock", type=str, required=True, help="stock name")
    parser.add_argument("--start", type=list, required=True, help="list representing start of date range")
    parser.add_argument("--end", type=list, required=True, help="list representing end of date range")

    args = parser.parse_args()
    print(generate_data(args.api_key_1,args.api_key_2, args.stock, args.start, args.end))

# Example usage:
# python Simple_Explanation.py --api_key_1 __ --api_key_2 __ --stock AAPL --start "2022-01-01" --end "2022-01-31"