import galai as gal
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model

    model = gal.load_model("standard")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    max_length = model_inputs.get('max_length', 60)
    top_p = model_inputs.get('top_p', None)
    new_doc = model_inputs.get('new_doc', False)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    output = model.generate(prompt,max_length=max_length,top_p=top_p,new_doc=new_doc)

    result = {"output": output}

    # Return the results as a dictionary
    return result
