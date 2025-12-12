# 🚀 Complete Walkthrough - DocuMind

This guide walks you through **every step** of setting up and using DocuMind, from installation to asking your first question.

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [First-Time Setup](#first-time-setup)
4. [Adding Your Documents](#adding-your-documents)
5. [Chatting with Your Documents](#chatting-with-your-documents)
6. [Understanding the Results](#understanding-the-results)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## 1. Prerequisites

Before you begin, ensure you have:
- **Python 3.10 or higher** installed
- **Git** installed (to clone the repository)
- At least **4GB of RAM** (8GB recommended)
- **Internet connection** (for initial model downloads)

**Check your Python version:**
```bash
python --version
```

---

## 2. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/TechSuvam/RAG_Demo.git
cd RAG_Demo
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**⏱️ Expected time:** 5-10 minutes (downloads ~2GB of packages)

**What gets installed:**
- LangChain (RAG framework)
- ChromaDB (Vector database)
- Transformers (AI models)
- Streamlit (Web interface)
- PDF/Markdown parsers

---

## 3. First-Time Setup

### Step 3: Prepare Your Data Folder

Create sample documents to test with:

**Option A: Use the provided samples**
The repository already includes:
- `data/sample.md` - About RAG concepts
- `data/python_intro.md` - About Python

**Option B: Add your own documents**
```bash
# Add any .md or .pdf files to the data folder
# Example:
cp ~/Documents/myfile.pdf data/
```

### Step 4: Launch the Application
```bash
streamlit run app.py
```

**What happens:**
1. Your browser opens automatically (http://localhost:8501)
2. The first time you run, it downloads AI models (~1GB)
   - `all-MiniLM-L6-v2` (Embedding model)
   - `google/flan-t5-base` (Language model)
3. This is a **one-time download** and takes 3-5 minutes

**You'll see:**
```
🧠 DocuMind
Chat with your documents (Markdown & PDF) purely offline.
```

---

## 4. Adding Your Documents

### Method 1: Upload via UI (Recommended)

1. **Look at the sidebar** (left side panel)
2. Click **"Browse files"** under "Upload Data"
3. Select one or more `.pdf` or `.md` files
4. Click **"Process Uploaded Files"**
5. Wait for "Knowledge Base Updated!" message

**Example:**
```
Sidebar:
├── Upload Data
│   ├── 📁 Browse files
│   └── ✅ Process Uploaded Files
└── Settings
    └── 🔄 Re-index Existing Data
```

### Method 2: Manual File Addition

1. Copy files directly to the `data/` folder
2. Click **"Re-index Existing Data"** in the sidebar
3. Wait for indexing to complete

---

## 5. Chatting with Your Documents

### Your First Question

**Example interaction:**

**You ask:** "What is RAG?"

**Behind the scenes:**
1. Your question is converted to numbers (embedding)
2. The system searches for relevant document chunks
3. It finds the 2 most similar pieces of text
4. An AI reads those chunks and generates an answer

**You receive:**
```
Answer: Retrieval-Augmented Generation (RAG) is a technique 
for enhancing the accuracy and reliability of generative AI 
models with facts fetched from external sources.

View Sources ▼
Source: ./data/sample.md
Content: RAG Concepts - Retrieval-Augmented Generation...
```

### Tips for Better Results

**✅ Good Questions:**
- "What are the main features of Python?"
- "Summarize the key points about machine learning"
- "How does RAG work?"

**❌ Questions That Won't Work Well:**
- "What happened yesterday?" (no context)
- "Tell me a joke" (not in your documents)
- Complex multi-step reasoning (the model is small)

---

## 6. Understanding the Results

### The Answer Section
The main response from the AI based on your documents.

### The Sources Section (Expandable)
Click **"View Sources"** to see:
- **Source File:** Which document the info came from
- **Content Preview:** The exact text that was used
- **Why this matters:** You can verify accuracy and learn more

**Example:**
```
View Sources ▼
Source: ./data/python_intro.md
Python Programming

Python is a high-level, interpreted programming 
language known for its readability...
```

### Chat History
All your previous questions and answers stay visible as you scroll up.

---

## 7. Troubleshooting

### Problem: "Vector DB Missing" Error
**Solution:** 
1. Click **"Re-index Existing Data"**
2. Or upload files via the sidebar

### Problem: App is Slow
**Causes:**
- First run (downloading models)
- Large PDFs being processed
- Underpowered CPU

**Solutions:**
- Wait for the initial download to complete
- Use smaller documents for testing
- Consider using a GPU-enabled server

### Problem: "I couldn't find any relevant information"
**Possible reasons:**
1. Your question is about topics not in your documents
2. The wording is too different from the document content
3. Try rephrasing the question

**Example:**
- ❌ "What's the deal with Python?"
- ✅ "What is Python used for?"

### Problem: Model Download Fails
**Solution:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
streamlit run app.py
```

---

## 8. Advanced Usage

### Running the CLI Version
For batch processing or testing:
```bash
python main.py
```

This runs pre-defined test queries without the UI.

### Adding More Documents Later
1. Upload new files via the sidebar
2. Click "Process Uploaded Files"
3. The new content is added to the existing knowledge base

### Customizing the AI Model

**To use a larger, smarter model** (requires more RAM):

Edit `app.py` line 27:
```python
# Change from:
model_id="google/flan-t5-base"

# To:
model_id="google/flan-t5-large"  # Needs 8GB+ RAM
```

### Using with GPU (Faster)
If you have an NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Quick Reference

| Action | Command/Button |
|--------|---------------|
| Start app | `streamlit run app.py` |
| Upload files | Sidebar → Browse files |
| Re-index | Sidebar → Re-index button |
| View sources | Click "View Sources" |
| Clear chat | Refresh the page |
| Stop app | Press `Ctrl+C` in terminal |

---

## 📊 Expected Performance

| Task | Time |
|------|------|
| Initial setup | 5-10 min |
| Model download | 3-5 min |
| Processing 1 PDF (10 pages) | 10-30 sec |
| Answering a question | 3-10 sec |

---

## 🔒 Privacy & Security

✅ **What's safe:**
- All processing happens on YOUR computer
- No data is sent to external servers
- Models run offline after initial download

⚠️ **What to know:**
- The `chroma_db/` folder contains your indexed data
- Don't share this folder if documents are sensitive
- Git is configured to ignore this folder

---

## 🎓 Next Steps

1. ✅ Test with the sample documents
2. ✅ Upload your own PDFs/Markdown files
3. ✅ Ask various questions
4. ✅ Check the sources for accuracy
5. ✅ Read the [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md) for technical details
6. ✅ Try deploying to AWS using [deployment.md](deployment.md)

---

## 💡 Pro Tips

1. **Organize your documents:** Group related files in the `data/` folder
2. **Use descriptive filenames:** Easier to track sources
3. **Start specific:** Ask narrow questions before broad ones
4. **Verify sources:** Always check the retrieved text
5. **Experiment:** Try different phrasings if results aren't good

---

**You're all set!** 🚀 Start chatting with your documents and enjoy your private AI assistant.

For technical questions, see [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)  
For deployment help, see [deployment.md](deployment.md)
