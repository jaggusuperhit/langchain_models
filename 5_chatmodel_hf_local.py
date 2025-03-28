from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = "D:/huggingface"

# Initialize the Hugging Face pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
    )
)

# Correctly initialize ChatHuggingFace with the 'llm' argument
model = ChatHuggingFace(llm=llm)

# Invoke the model with a question
result = model.invoke("What is the capital of India?")

# Print the result
print(result.content)
