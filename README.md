## Private Q&A Chatbot

This repository contains a private question and answer (Q&A) chatbot powered by Streamlit. The chatbot leverages language embeddings and a pre-trained language model to provide accurate answers to user queries based on a collection of documents.

### Installation

To install the required dependencies, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Usage

1. Clone this repository to your local machine.
2. Install the dependencies as mentioned above.
3. Ensure you have set up your environment variables, particularly `OPENAI_API_KEY`, which is required for the OpenAI language model.
4. Prepare your documents in a directory. The default directory is set to "./data". Ensure your documents are in text format.
5. Run the `app.py` script:

```bash
streamlit run app.py
```

6. Access the chatbot through your web browser.

### How it Works

- The chatbot utilizes FAISS, an efficient similarity search library, to index document embeddings.
- Documents are first split into smaller chunks for efficient processing.
- Language embeddings are generated for each document chunk using OpenAI's language model.
- The FAISS index is populated with the document embeddings.
- Upon receiving a user query, the chatbot retrieves the most relevant documents from the index and passes them through a question-answering chain.
- The chatbot returns the most appropriate answer based on the user query.

Feel free to contribute to this project by submitting pull requests or opening issues. For major changes, please open an issue first to discuss the proposed changes.