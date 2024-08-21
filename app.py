import os
from dotenv import load_dotenv
import lamini
from lamini import Lamini

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
lamini.api_key = os.getenv("LAMINI_API_KEY")

llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

def get_data():
    data = [
        {
            "input": "Are there any step-by-step tutorials or walkthroughs available in the documentation?",
            "output": "Yes, there are step-by-step tutorials and walkthroughs available in the documentation section. Here’s an example for using Lamini to get insights into any python SDK: https://lamini-ai.github.io/example/",
        },
        # ... (rest of your data)
    ]
    return data

data = get_data()

llm.tune(data_or_dataset_id=data,
         finetune_args={'learning_rate': 1.0e-4}
         )

'''
Common hyperparameters to tune include:

learning_rate (float) - the learning rate of the model
early_stopping (bool) - whether to use early stopping or not
max_steps (int) - the maximum number of steps to train for
optim (str) - the optimizer to use, e.g. adam or sgd, a string from HuggingFace
'''
