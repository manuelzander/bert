import logging

# from multiprocessing import Pool
import os

import wikipedia
from config import ROOT_DIR, MODEL_NAME, NO_ARTICLES, NO_SENTENCES
from flask import Flask, render_template
from flask_socketio import SocketIO
from modelling import repl

# Flask initialization
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
app.logger.setLevel(logging.DEBUG)
socketio = SocketIO(app, async_mode=None)


# def summary(r):
#     return wikipedia.summary(r)


def ask_wiki(question: str):
    results = wikipedia.search(question, results=NO_ARTICLES)
    app.logger.info("Wikipedia articles: %s", results)

    summaries = [wikipedia.summary(result, sentences=NO_SENTENCES) for result in results]

    # with Pool(processes=TOP_N) as pool:
    #     summaries = pool.map(
    #         func=summary,
    #         iterable=results[:TOP_N]
    #     )

    combined = " ".join(summaries)
    app.logger.info("SUCCESS - Wikipedia summaries combined")
    return combined


@app.route("/")
def index():
    return render_template("index.html")


def answer_sent():
    app.logger.info("SUCCESS - answer sent")


@socketio.on("my_question")
def handle_my_question(data):

    # Get context
    app.logger.info("SUCCESS - question received: " + str(data["question"]))
    app.logger.info("Attempting to create context...")

    try:
        context = ask_wiki(data["question"])
    except Exception as e:
        app.logger.error("ERROR - couldn't create context due to: %s", e)
        context = "No context created"

    # Shorten context
    context = context.split(" ")
    magic_number = 384 - len(data["question"].split(" ")) - 3
    context = context[:magic_number]
    data["context"] = " ".join(context)

    app.logger.info("SUCCESS - context created: " + str(data["context"]))
    app.logger.info("Attempting to get answer...")

    # Get answer
    try:
        data["answer"] = repl.ask(model, tokenizer, data["question"], data["context"])
    except Exception as e:
        app.logger.error("ERROR - couldn't get answer due to: %s", e)
        data["answer"] = "No answer found"

    app.logger.info("SUCCESS - answer found: " + str(data["answer"]))
    socketio.emit("Response", data, callback=answer_sent())


if __name__ == "__main__":
    app.logger.info("Loading model...")
    model_path = os.path.join(ROOT_DIR, MODEL_NAME)
    model, tokenizer = repl.get_model(model_path)
    app.logger.info("Starting server...")
    socketio.run(app, debug=True)
