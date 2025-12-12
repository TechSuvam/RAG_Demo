# Fix Explanation: Improving LLM Responses

## 🐛 The Problem
When you sent the message **"Hello"**, the RAG system retrieved the most "relevant" documents it could find (which was the Python file). 
Since the AI model (`flan-t5-base`) is trained to **answer questions based STRICTLY on the provided context**, it got confused. It didn't find a greeting in the "Python" text, so it panicked and just spat out the content of the document itself as the "answer".

This is a common issue with basic RAG pipelines called **Context Leakage** or **Hallucination**.

## 🛠️ The Fix (Prompt Engineering)
I updated the logic in `app.py` to give the AI **stricter instructions**.

### Before
The prompt was too simple. It just said:
> "Use the context below to answer the question."

### After (The Fix)
I changed the prompt to establish **rules**:

```python
full_prompt = f"""
Use the following pieces of context to answer the question at the end. 

1. If the answer is not in the context, just say that you don't know. (Prevents making things up)
2. If the question is a greeting (like hello, hi), simply greet the user back. (Handles social chit-chat)

Context:
{context}

Question: {prompt}

Helpful Answer:
"""
```

## ✅ Result
Now, when you say "Hello":
1. The system still retrieves documents (because that's how RAG works).
2. BUT the AI reads the new instructions.
3. It sees rule #2: *"If the question is a greeting... greet the user back"*.
4. Instead of reading the Python file aloud, it simply says "Hello!" or similar.
