# This code does the following:
# 1. It reads the documents from the data folder and then loads the documents into the VectorStoreIndex.
# 2. Creates a PromptTemplate object with the prompt string.
# 3. Creates an Vector Index Retriever object with the index and similarity_top_k=5. (Data ingested from folder)
# 4. Creates a QueryPipeline object and adds the llm, prompttemplate, reranker, and summarizer modules.
# 5. Adds links between the modules to build the pipeline sequence.
# 6. Runs the pipeline 

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
import os 

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# pipenv install llama-index-postprocessor-cohere-rerank
# Go to cohere.com and create your API, set up COHERE_API_KEY in your .env file

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)

# print the number of documents
print(f"Number of documents: {len(documents)}")

# define modules
prompt_str = "What are the potential Risks of Harm from {topic}?"
prompt_tmpl = PromptTemplate(prompt_str)

llm = OpenAI(model="gpt-3.5-turbo")

retriever = index.as_retriever(similarity_top_k=5)  # configure retriever with top_k=5

# Rerank provides a powerful semantic boost to the search quality of any keyword or vector search
# system without requiring any overhaul or replacement
reranker = CohereRerank()   

# TreeSummerize: When provided with text nodes and a query, recursively merges them (like a tree - bottoms up) 
# and returns the root node as the result. 
summarizer = TreeSummarize(llm=llm)

# define query pipeline
p = QueryPipeline(verbose=True)

# add modules (LLM, prompt template, retreiver object, summarizer and reranker) to the pipeline
p.add_modules(
		{
    	"llm": llm,
    	"prompt_tmpl": prompt_tmpl,
    	"retriever": retriever,
    	"summarizer": summarizer,
    	"reranker": reranker
	}
)

# add links between the modules to create the sequence of the pipeline
p.add_link("prompt_tmpl", "llm")   
p.add_link("llm", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")   
p.add_link("llm", "reranker", dest_key="query_str")     
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

# run the pipeline
output = p.run(topic="Content Exposure")

print(output)
