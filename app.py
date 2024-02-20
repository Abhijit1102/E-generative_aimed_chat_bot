from flask import Flask, render_template, jsonify, request

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

from src.utils import download_hugging_face_embeddings
from src.prompt import prompt_template

from src.logger import logging
from src.exception import CustomException

import chromadb

app = Flask(__name__)

try:
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path="data/")
    collection = client.get_collection(name="medchat")

    # Initialize PromptTemplate, CTransformers, and embeddings
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens': 512, 'temperature': 0.5})
    embeddings = download_hugging_face_embeddings()

    def embedding_query(query):
        result = collection.query(
            query_embeddings=embeddings.embed_query(query),
            n_results=3
        )
        context = result["documents"][0]
        return context

    llm_chain = LLMChain(prompt=PROMPT, llm=llm)

except Exception as e:
    logging.error(f"Error during initialization: {e}")
    raise CustomException(e)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        question = request.form["msg"]

        context = embedding_query(question)

        inputs = {"context": context, "question": question}

        result = llm_chain(inputs)
        logging.info(f"Response: {result['text']}")
        return str(result["text"])

    except Exception as e:
        logging.error(f"Error during chat processing: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
