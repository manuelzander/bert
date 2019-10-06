# BERT HACKATHON
Bert QA project for internal hackathon
# Environment setting
1. Clone the repo to your local machine
2. Create environment from yaml file by running the command:
```conda env create -f env.yml```
3. Activate the new environment in conda:
```source activate bert-qa-hackathon```
4. Install PyTorch in the new environment, install instruction -> https://pytorch.org/

It's really a pain to include the torch in the env.yml. 
# Download the model
The pre-trained model object can be downloaded from:
https://drive.google.com/open?id=1UbhMvtUeX1LiRA9uDiontnOgqps1ooy5

Please unzip it and replace under ./bert_hackathon/flask/ 
# Run the web service
1. change the line 113 in ./bert_hackathon/flask/modelling/repl.py to point to the location of the unzipped Bert folder from previous step
2. run ```python main.py``` from its directory level

