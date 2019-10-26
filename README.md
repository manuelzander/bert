# BERT Q&A engine

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Clone the repo to your local machine

Create a virtual environment for Python 3 with:

```
virtualenv -p python3 env
```

Activate the virtual environment with:

```
source env/bin/activate
```

### Installing packages

Then the used Python libraries/packages can be installed with:

```
pip install -r requirements.txt
```

## Download the model

The pre-trained model object can be downloaded from:

https://drive.google.com/open?id=1UbhMvtUeX1LiRA9uDiontnOgqps1ooy5

Please unzip it and replace under ./bert/flask/

## Run the flask server

Change the line 113 in ./bert/flask/modelling/repl.py to point to the location of the unzipped Bert folder from previous step

Run ```python main.py``` inside ./bert/flask

