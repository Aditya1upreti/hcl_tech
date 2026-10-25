# 90-Second Demo Script

## Opening (0-10 seconds)
"Hello! Welcome to the RAG-Based Document Q&A System. I'm going to show you how this AI-powered system lets you query policy documents using natural language."

## Step 1: Starting the System (10-20 seconds)
"First, let me start the application. I'll run the ingestion pipeline to load our policy documents, then start the Streamlit UI."

**Commands displayed on screen:**
```bash
pip install -r requirements.txt
python src/ingest.py
streamlit run app.py
```

## Step 2: System Initialization (20-35 seconds)
"The system starts up. Notice the sidebar - it shows the provider status. Since we're running without a Gemini API key, the system is in retrieval-only mode. The key status indicator is red."

**Visual:** Sidebar showing "✗ No API key - retrieval-only mode" and "retrieval-only" as the provider.

## Step 3: Asking a Question (35-55 seconds)
"I'll now ask a question about our policies. Let me ask: 'What are the rules for taking approved leave according to the leave policy?'"

**Visual:** User types in the chat input, press Enter.

"The system retrieves the top-4 most relevant chunks from the policy documents. You can see the retrieval scores - these show how similar each chunk is to your query. The system displays: 'File (page X) — similarity: Y%' for each source."

**Visual:** Expander panel opens showing source chunks with file names, page numbers, and similarity percentages.

## Step 4: Answer Display (55-70 seconds)
"Since we're in retrieval-only mode, the system displays the retrieved source chunks instead of generating an answer. It says: 'I don't know based on the provided documents.' But importantly, it shows us the source material so we can verify the information ourselves."

**Visual:** Answer area shows "I don't know based on the provided documents." Expander below shows 4 source chunks.

## Step 5: Verification and Value (70-80 seconds)
"See how the system displays the actual text from the leave policy? This is the key value proposition - you get to verify any answer against the original source document. The source chunks show: the file name (leave_policy.docx), the page number, and the similarity score."

**Visual:** Highlight the source chunk information displayed beside the answer.

## Step 6: Adding a Document (80-85 seconds)
"Let me show you how to add a new document. I'll upload a new policy file through the sidebar file uploader, click 'Ingest Uploaded Documents', and the system processes it immediately."

**Visual:** Sidebar file uploader, button click, ingestion progress, success message.

## Step 7: Changing Providers (85-90 seconds)
"When you have a Gemini API key, you can add it to the .env file and switch the provider in the sidebar. The system will then use Gemini to generate answers instead of just showing source chunks."

**Visual:** Sidebar provider radio changed from "retrieval-only" to "gemini", key status turns green.

## Closing (85-90 seconds)
"That's the RAG-Based Document Q&A System in a nutshell! It works without an API key by showing you source documents, and when you have a key, it can generate answers. Thank you for watching!"

**Final command on screen:**
```bash
# Get Gemini key at: aistudio.google.com
# Add GEMINI_API_KEY to .env and restart: streamlit run app.py