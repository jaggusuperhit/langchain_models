from langchain_huggingface import HuggingFaceEmbeddings

# Initialize with the correct parameter name
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of France?"

# Generate embedding vector
vector = embedding.embed_query(text)

# Print the embedding vector
print(str(vector))
