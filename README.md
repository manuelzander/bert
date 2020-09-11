# BERT Q&A engine

![status](https://img.shields.io/github/workflow/status/manuelzander/bert/Python%20application/master?label=actions&logo=github&style=for-the-badge) ![last-commit](https://img.shields.io/github/last-commit/manuelzander/bert/master?logo=github&style=for-the-badge) ![issues-pr-raw](https://img.shields.io/github/issues-pr-raw/manuelzander/bert?label=open%20prs&logo=github&style=for-the-badge) [![license](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## Prerequisites

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

The code is run and tested with Python 3.7.4 on macOS 10.14.6.

### Environment

Clone the repo to your local machine.

Create a virtual environment for Python 3 with:

```
virtualenv -p python3 env
```

Activate the virtual environment with:

```
source env/bin/activate
```

Install the required Python packages with:

```
pip install -r requirements.txt
```

You will also need https://github.com/huggingface/transformers.

Within the ```source``` folder, run:

```
git clone git@github.com:huggingface/transformers.git
```

### Model

The pre-trained model object can be downloaded from:

https://drive.google.com/open?id=1UbhMvtUeX1LiRA9uDiontnOgqps1ooy5

Create a folder ```source/bert``` and unzip the model in there.

## Running the flask server

Run ```python3 source/main.py``` and open http://127.0.0.1:5000/

If you just want to pass a question and a context to the model, run:

```
python3 source/modelling/repl.py --question question --context context
```
