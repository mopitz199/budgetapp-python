from fastapi import FastAPI 
from os import environ as env

from openai import OpenAI
from pydantic import BaseModel
from decimal import Decimal
from datetime import date

class Transaction(BaseModel):
    date: date
    description: str
    amount: Decimal

class BankTransactionExtractionFormat(BaseModel):
    transactions: list[Transaction]

class AnalayzeBankTransactionRequest(BaseModel):
    image_url: str

app = FastAPI()

@app.post("/analyze-bank-transactions")
def analyze_bank_transactions(request: AnalayzeBankTransactionRequest):
    client = OpenAI(api_key=env["OPENAI_KEY"])

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "Behave as an expert extracting bank statements transaction. Understanding how to format the date, description and amount of each bank transaction",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract the transactions from this bank statement"},
                    {
                        "type": "input_image",
                        "image_url": request.image_url
                    }
                ]
            }
        ],
        text_format=BankTransactionExtractionFormat,
    )
    
    return {"details": response.output_parsed.transactions}