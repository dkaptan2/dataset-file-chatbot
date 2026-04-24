# Dataset Q&A Chatbot

A file-based question answering system that combines lightweight retrieval with language-model reasoning to help users explore uploaded datasets through natural-language queries.

This project supports **CSV, Excel, JSON, and TXT** inputs. It uses a retrieval-first pipeline to identify the most relevant records from the uploaded file, then passes those records to a language model to generate a grounded response.

<img width="1575" height="862" alt="Screenshot 2026-04-24 124443" src="https://github.com/user-attachments/assets/8e4c9114-00dc-4442-b856-d3b69376565c" />

## Overview

The system is designed as a simple retrieval-augmented question answering workflow:

1. The user uploads a file
2. The file is parsed into searchable text records
3. The user asks a natural-language question
4. The app retrieves the most relevant records using **TF-IDF vectorization** and **cosine similarity**
5. The retrieved context is sent to a language model
6. The model returns a grounded answer based only on the retrieved records

This approach keeps the system efficient, interpretable, and much more reliable than sending an entire file directly to a model.

## Features

- Upload and analyze:
  - `.csv`
  - `.xlsx`
  - `.xls`
  - `.json`
  - `.txt`
- Natural-language question answering over uploaded files
- Retrieval-based grounding using TF-IDF and cosine similarity
- LLM-powered answer synthesis using retrieved context only
- Simple local web interface built with Gradio

## High-Level Architecture

```text
User File + Question
        ↓
File Parsing / Record Extraction
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity Retrieval
        ↓
Top Matching Records
        ↓
LLM Answer Synthesis
        ↓
Grounded Final Answer

## How It Works

1. File Parsing

Uploaded files are converted into a list of searchable text records.

CSV / Excel rows are flattened into text strings
JSON objects are flattened into key-value text
TXT files are split into smaller searchable chunks

2. TF-IDF Retrieval

The app uses TF-IDF (Term Frequency-Inverse Document Frequency) to numerically represent:

-the user’s question
-the extracted file records

This creates a lightweight retrieval system that can compare the question against the uploaded data.

3. Cosine Similarity Ranking

After vectorization, cosine similarity is used to rank the file records by relevance to the user’s question.

The top-ranked records are selected as the supporting context.

4. Grounded LLM Response

Instead of sending the entire file to a model, the app sends:

-the user’s question
-only the top retrieved records

This helps keep the answer grounded in the uploaded file and reduces irrelevant output.

Tech Stack:
-Python
-Gradio for the UI
-Pandas for file loading and preprocessing
-scikit-learn
-TfidfVectorizer
-cosine_similarity
-Requests for API communication
-OpenRouter API for model inference

## Setup Instructions:
Step 1: Clone the repository
git clone https://github.com/YOUR_USERNAME/dataset-file-chatbot.git
cd dataset-file-chatbot

Replace YOUR_USERNAME with your actual GitHub username.

Step 2: Install dependencies
python -m pip install -r requirements.txt

If that does not work, try:

py -m pip install -r requirements.txt

Step 3: Set your OpenRouter API key

You need an OpenRouter API key stored as an environment variable before running the app.

PowerShell
$env:OPENROUTER_API_KEY="your_api_key_here"

Command Prompt
set OPENROUTER_API_KEY=your_api_key_here

Mac / Linux
export OPENROUTER_API_KEY="your_api_key_here"

Important: set the API key in the same terminal window where you will run the app.

Step 4: Run the app
python app.py

If that does not work, try:

py app.py
Step 5: Open the local web app

After running the app, the terminal should display a local URL, usually:

http://127.0.0.1:7860

Open that link in your browser.

Fast Setup Commands:

PowerShell

python -m pip install -r requirements.txt

$env:OPENROUTER_API_KEY="your_api_key_here"

python app.py

Command Prompt

python -m pip install -r requirements.txt

set OPENROUTER_API_KEY=your_api_key_here

python app.py

Mac / Linux

python -m pip install -r requirements.txt

export OPENROUTER_API_KEY="your_api_key_here"

python app.py

Example Use Cases:
-Ask questions about course catalogs
-Explore customer complaint datasets
-Query JSON records without manually reading each entry
-Search through uploaded text-heavy files
-Ask high-level questions about structured tabular data
