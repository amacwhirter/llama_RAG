# Example: Explicitly configure context_window and num_output
# This is in case you need to explicitly configure the context_window and num_output (limit the output size)

# Install the llama-index-readers-file package before running this program
# !pipenv install llama-index-readers-file

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

documents = SimpleDirectoryReader("data").load_data()

# set context window
# this is done to limit the input size and avoid the error "Input too large"
context_window = 4096
# set number of output tokens
# this is done to limit the output size
num_output = 200 

# define LLM
Settings.llm = OpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    max_tokens=num_output,
    context_window=context_window,
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)
# print the number of documents
print(f"Number of documents: {len(documents)}")

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer - it turns the response Data into a human-readable format
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,   
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
response = query_engine.query("What The Potential Benefits of Social Media Use Among Children and Adolescents?") 
print(response)
