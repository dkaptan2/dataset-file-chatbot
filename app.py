import os
import json
import pandas as pd
import gradio as gr
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def chunk_text(text, chunk_size=700, overlap=120):
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks


def load_items_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path).fillna("")
        return [
            " | ".join(f"{col}: {row[col]}" for col in df.columns)
            for _, row in df.iterrows()
        ]

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path).fillna("")
        return [
            " | ".join(f"{col}: {row[col]}" for col in df.columns)
            for _, row in df.iterrows()
        ]

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = []
            for item in data:
                if isinstance(item, dict):
                    items.append(" | ".join(f"{k}: {v}" for k, v in item.items()))
                else:
                    items.append(str(item))
            return items

        if isinstance(data, dict):
            return chunk_text(json.dumps(data, indent=2, ensure_ascii=False))

        return [str(data)]

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return chunk_text(f.read())

    raise ValueError("Unsupported file type. Please upload CSV, XLSX, JSON, or TXT.")


def retrieve_relevant_items(items, question, top_k=4):
    items = [item for item in items if str(item).strip()]
    if not items:
        return []

    if len(items) > 2000:
        items = items[:2000]

    vectorizer = TfidfVectorizer(stop_words="english")
    item_vectors = vectorizer.fit_transform(items)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, item_vectors).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]
    return [(items[i], float(scores[i])) for i in top_indices]


def call_openrouter(question, retrieved_items):
    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY. Set it in your environment first."

    context_blocks = []
    for idx, (item, score) in enumerate(retrieved_items, start=1):
        context_blocks.append(
            f"[Record {idx} | similarity={score:.3f}]\n{item}"
        )

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a dataset question-answering assistant. "
        "Answer ONLY using the provided records. "
        "If the answer is not supported by the records, say that clearly. "
        "Be concise and cite the relevant record numbers in your answer."
    )

    user_prompt = f"""User question:
{question}

Retrieved records:
{context_text}

Instructions:
- Answer using only the retrieved records.
- If multiple records support the answer, mention them.
- If the records are insufficient, say so.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"OpenRouter API error: {e}"
    except (KeyError, IndexError):
        return f"Unexpected API response: {response.text}"


def answer_question(file_path, question):
    if not file_path:
        return "Please upload a dataset file first."

    if not question or not question.strip():
        return "Please type a question."

    try:
        items = load_items_from_file(file_path)
    except Exception as e:
        return f"Error reading file: {e}"

    if not items:
        return "I could not find any usable content in that file."

    try:
        retrieved = retrieve_relevant_items(items, question, top_k=4)
    except Exception as e:
        return f"Retrieval error: {e}"

    if not retrieved:
        return "No relevant records found."

    final_answer = call_openrouter(question, retrieved)

    supporting_records = "\n\n".join(
        f"**Record {i}** (similarity {score:.3f})\n{item}"
        for i, (item, score) in enumerate(retrieved, start=1)
    )

    return (
        f"# Final Answer\n\n{final_answer}\n\n"
        f"---\n\n# Supporting Retrieved Records\n\n{supporting_records}"
    )


demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload dataset", type="filepath"),
        gr.Textbox(
            label="Ask a question",
            placeholder="Example: Which entries mention shipping delays?"
        ),
    ],
    outputs=gr.Markdown(label="Answer"),
    title="Dataset Q&A with Free Model API",
    description="Upload a CSV, Excel, JSON, or TXT file and ask a grounded question about it.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()