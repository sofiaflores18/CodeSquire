pip install -U "huggingface_hub[cli]"
hf auth login
hf repo create Adammouedden/codesquire
hf upload Adammouedden/codesquire ./model
