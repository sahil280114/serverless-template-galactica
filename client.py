import banana_dev as banana
import os

api_key = os.environ.get("API_KEY")
model_key = os.environ.get("MODEL_KEY")

model_inputs = {'prompt': 'Scaled dot product attention:\n\n\\[',"max_length":60,"top_p":0.7,"new_doc":False}

model_outputs = banana.run(api_key, model_key, model_inputs)

print(model_outputs)
