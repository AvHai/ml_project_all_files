from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import os
from typing import Optional, List
import json
from io import StringIO
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize FastAPI app
app = FastAPI(title="Flight Information Chatbot with RAG", version="2.0.0")

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBMxt3eimaVEqscA8l0RRhtNTcBX9ZU34Q"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

# Global variables for data and vector store
flight_data = None
csv_columns = []
vectorizer = None
flight_vectors = None
flight_texts = []

class ChatRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    confidence: str
    relevant_flights: List[dict]

class UploadResponse(BaseModel):
    message: str
    columns: list
    row_count: int

def create_flight_text(row):
    """Convert flight row to searchable text"""
    text_parts = []
    for col, val in row.items():
        if pd.notna(val):
            text_parts.append(f"{col}: {val}")
    return " | ".join(text_parts)

def preprocess_text(text):
    """Clean and normalize text for better matching"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vector_store():
    """Build TF-IDF vector store from flight data"""
    global flight_data, vectorizer, flight_vectors, flight_texts
    
    if flight_data is None:
        return False
    
    # Create searchable text for each flight
    flight_texts = []
    for _, row in flight_data.iterrows():
        flight_text = create_flight_text(row)
        flight_texts.append(preprocess_text(flight_text))
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    flight_vectors = vectorizer.fit_transform(flight_texts)
    
    return True

def search_relevant_flights(query, max_results=5):
    """Search for relevant flights using vector similarity"""
    global vectorizer, flight_vectors, flight_data
    
    if vectorizer is None or flight_vectors is None:
        return []
    
    # Process query
    query_processed = preprocess_text(query)
    query_vector = vectorizer.transform([query_processed])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, flight_vectors).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-max_results:][::-1]
    
    relevant_flights = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            flight_info = flight_data.iloc[idx].to_dict()
            flight_info['similarity_score'] = float(similarities[idx])
            relevant_flights.append(flight_info)
    
    return relevant_flights

@app.get("/")
async def root():
    return {"message": "Flight Information Chatbot API with RAG", "status": "running"}

@app.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process a CSV file containing flight information."""
    global flight_data, csv_columns
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        
        # Parse CSV
        flight_data = pd.read_csv(StringIO(csv_string))
        csv_columns = flight_data.columns.str.strip().tolist()
        flight_data.columns = flight_data.columns.str.strip()
        
        # Build vector store
        if not build_vector_store():
            raise HTTPException(status_code=500, detail="Failed to build vector store")
        
        return UploadResponse(
            message="CSV uploaded and vector store built successfully",
            columns=csv_columns,
            row_count=len(flight_data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user questions about flight information using RAG."""
    global flight_data, csv_columns, model
    
    if flight_data is None:
        raise HTTPException(status_code=400, detail="No CSV data uploaded. Please upload a CSV file first.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # Search for relevant flights
        relevant_flights = search_relevant_flights(request.question, request.max_results)
        
        if not relevant_flights:
            return ChatResponse(
                answer="I don't know. I couldn't find any relevant flight information for your question in the available data.",
                confidence="low",
                relevant_flights=[]
            )
        
        # Prepare context with only relevant flights
        context_flights = []
        for flight in relevant_flights:
            # Remove similarity score for context
            flight_copy = {k: v for k, v in flight.items() if k != 'similarity_score'}
            context_flights.append(flight_copy)
        
        # Create focused prompt with limited context
        prompt = f"""
You are a flight information assistant. Answer the user's question based on the relevant flight data provided.

AVAILABLE DATA COLUMNS: {', '.join(csv_columns)}

RELEVANT FLIGHTS (most similar to the query):
{json.dumps(context_flights, indent=2, default=str)}

USER QUESTION: {request.question}

INSTRUCTIONS:
1. Answer the user's question using ONLY the relevant flights provided above
2. Be specific and cite actual flight number and time of departure and arrival only give 3 options
3. If the provided flights don't contain the information needed to answer the question, respond with "I don't know" and explain what information would be needed
4. Keep your response concise and helpful
5. Use natural language, not technical jargon

Answer:"""

        # Generate response using Gemini with limited context
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Determine confidence
        if "I don't know" in answer or "not available" in answer.lower():
            confidence = "low"
        elif len(relevant_flights) >= 3:
            confidence = "high"
        else:
            confidence = "medium"
        
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            relevant_flights=relevant_flights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/search-flights")
async def search_flights(request: ChatRequest):
    """Direct search for flights without AI processing (faster alternative)"""
    global flight_data
    
    if flight_data is None:
        raise HTTPException(status_code=400, detail="No CSV data uploaded")
    
    try:
        relevant_flights = search_relevant_flights(request.question, request.max_results)
        
        return {
            "query": request.question,
            "found_flights": len(relevant_flights),
            "flights": relevant_flights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching flights: {str(e)}")

@app.get("/data-info")
async def get_data_info():
    """Get information about the currently loaded CSV data."""
    global flight_data, csv_columns
    
    if flight_data is None:
        return {"message": "No data loaded", "columns": [], "row_count": 0}
    
    return {
        "message": "Data loaded successfully",
        "columns": csv_columns,
        "row_count": len(flight_data),
        "vector_store_ready": vectorizer is not None,
        "sample_data": flight_data.head(3).to_dict('records')
    }

@app.delete("/clear-data")
async def clear_data():
    """Clear the currently loaded CSV data."""
    global flight_data, csv_columns, vectorizer, flight_vectors, flight_texts
    
    flight_data = None
    csv_columns = []
    vectorizer = None
    flight_vectors = None
    flight_texts = []
    
    return {"message": "Data and vector store cleared successfully"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_configured": model is not None,
        "data_loaded": flight_data is not None,
        "vector_store_ready": vectorizer is not None
    }

# Advanced search endpoint for testing vector similarity
@app.post("/debug-search")
async def debug_search(request: ChatRequest):
    """Debug endpoint to see similarity scores"""
    global vectorizer, flight_vectors, flight_data
    
    if flight_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        query_processed = preprocess_text(request.question)
        query_vector = vectorizer.transform([query_processed])
        similarities = cosine_similarity(query_vector, flight_vectors).flatten()
        
        # Get all results with scores
        results = []
        for idx, score in enumerate(similarities):
            if score > 0.05:  # Lower threshold for debugging
                flight_info = flight_data.iloc[idx].to_dict()
                results.append({
                    "flight": flight_info,
                    "similarity": float(score),
                    "text_representation": flight_texts[idx]
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "query": request.question,
            "processed_query": query_processed,
            "total_matches": len(results),
            "top_matches": results[:10]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)