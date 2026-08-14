import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
load_dotenv()

app = Flask(__name__)

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}")
    ]
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        user_input = request.form["question"]
        response = chain.invoke({"question": user_input})
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)