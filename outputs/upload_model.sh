pip install -U "huggingface_hub[cli]"
hf auth login
hf repo create Adammouedden/squire
hf upload Adammouedden/squire ./model
