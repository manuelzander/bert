import logging
from multiprocessing import Pool
import os
import flask
import wikipedia
from config import ROOT_DIR, MODEL_NAME
from flask import render_template
from flask_socketio import SocketIO
from modelling import repl

TOP_N = 3
NUM_SENT = 10

# Flask initialization
app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.logger.setLevel(logging.DEBUG)
socketio = SocketIO(app, async_mode=None)

# Loading the model
model_path = os.path.join(ROOT_DIR, MODEL_NAME)
model, tokenizer = repl.get_model(model_path)


def summary(r):
    return wikipedia.summary(r, sentences=NUM_SENT)


def ask_wiki(question: str):
    results = wikipedia.search(question)
    print(results)

    with Pool(processes=TOP_N) as pool:
        summaries = pool.map(
            func=summary,
            iterable=results[:TOP_N]
        )

    combined = " ".join(summaries)
    print(summaries)
    return combined


@app.route("/")
def index():
    return render_template('index.html')


def answer_sent():
    app.logger.info('Answer sent...')


@socketio.on('my_question')
def handle_my_question(data):

    # get context
    context = ask_wiki(data["question"])
    context = context.split(" ")

    # shorten it
    magic_number = 384 - len(data["question"].split(" ")) - 3
    context = context[:magic_number]

    data["context"] = " ".join(context)

    app.logger.info('Question and context received...')
    app.logger.info('Question: ' + str(data['question']))
    app.logger.info('Context: ' + str(data['context']))

    data['answer'] = repl.ask(model, tokenizer, data["question"], data["context"])

    app.logger.info('Answer: ' + str(data['answer']))
    socketio.emit('Response', data, callback=answer_sent())


if __name__ == '__main__':
    app.logger.info("Start server...")
    socketio.run(app, debug=True)
