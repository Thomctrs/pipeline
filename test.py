from langchain_community.embeddings import FastEmbedEmbeddings

supported_models = FastEmbedEmbeddings.list_supported_models()
print("Supported models:", supported_models)
