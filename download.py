# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPTJ model

import galai as gal
import torch

def download_model():
    # do a dry run of getting model path, which downloads the weights into the correct path if not present
    path = gal.get_checkpoint_path("base")
    tok_path = gal.get_tokenizer_path()

if __name__ == "__main__":
    download_model()