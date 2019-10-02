import logging

import flask
from flask import render_template
from flask_socketio import SocketIO

from modelling import repl

# Flask initialization
app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.logger.setLevel(logging.DEBUG)
socketio = SocketIO(app, async_mode=None)

# Loading the model
model_path = './bert'
model, tokenizer = repl.get_model(model_path)


@app.route("/")
def index():
    return render_template('index.html')


def answer_sent():
    app.logger.info('Answer sent...')


@socketio.on('my_question')
def handle_my_question(data):
    app.logger.info('Question and context received...')
    app.logger.info('Question: ' + str(data['question']))
    data['context'] = 'sample_context'
    app.logger.info('Context: ' + str(data['context']))
    data['answer'] = repl.ask(model, tokenizer, data["question"], data["context"])
    app.logger.info('Answer: ' + str(data['answer']))
    socketio.emit('Response', data, callback=answer_sent())


if __name__ == '__main__':
    app.logger.info("Start server...")
    socketio.run(app, debug=True)
