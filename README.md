# BERT Q&A engine

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

### Model

The pre-trained model object can be downloaded from:

https://drive.google.com/open?id=1UbhMvtUeX1LiRA9uDiontnOgqps1ooy5

Create a folder ```flask/bert``` and unzip the model in there.

## Running the flask server

Change line 113 in ```flask/modelling/repl.py``` to point to the location of the unzipped model folder from previous step.

Run ```python flask/main.py``` and open http://127.0.0.1:5000/

