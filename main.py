import json
from datetime import date, datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header
from os import environ as env

# Firebase
import firebase_admin
from firebase_admin import credentials, auth, firestore
# Open AI
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
    images_url: list[str]

app = FastAPI()
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)


def verify_firebase_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="not authorized")

    id_token = auth_header.split(" ")[1]

    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token  # puedes extraer uid, email, etc.
    except Exception as e:
        raise HTTPException(status_code=401, detail="invalid token")
    
def build_request_input(images_url):
    content = [
        {
            "type": "input_text",
            "text": "Extract the transactions from this bank statement and reflect all the withdrawals with a negative number and the deposit with a positive number"
        }
    ]
    for image_url in images_url:
        content.append(
            {
                "type": "input_image",
                "image_url": image_url,
            }
        )

    input = [
        {
            "role": "system",
            "content": "Behave as an expert extracting bank statements transaction. Understanding how to format the date, description and amount of each bank transaction",
        },
        {
            "role": "user",
            "content": content
        }
    ]
    return input

@app.post("/analyze-bank-transactions")
def analyze_bank_transactions(
    request: AnalayzeBankTransactionRequest,
    authorization: str = Header(None)
):
    token_extrated_data = verify_firebase_token(authorization)
    user_id=token_extrated_data.get("user_id")
    email_verified=token_extrated_data.get("email_verified")
    print(token_extrated_data)

    if not email_verified:
        raise HTTPException(status_code=401, detail="email not verified")


    db = firestore.client()
    client = OpenAI(api_key=env["OPENAI_KEY"])

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=build_request_input(request.images_url),
        text_format=BankTransactionExtractionFormat,
    )

    analysis_requirement_response = {
        "created_at": datetime.now(timezone.utc).timestamp(),
        "images_url": request.images_url,
        "user_id": user_id,
        "response_id": response.id,
        "status": response.output[0].status,
        "error": response.error.message if response.error else response.error,
        "incomplete_details": response.incomplete_details.reason if response.incomplete_details else response.incomplete_details,
        "model": response.model,
        "json_output": json.loads(response.output[0].content[0].text),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
    ref = db.collection("analysis_requirement").add(analysis_requirement_response)

    return {"details": response.output_parsed.transactions}
