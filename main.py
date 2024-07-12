from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings


model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}

gpt4embedding = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=gpt4embedding)

question = "What is Chain of Thought?"
docs = vectorstore.similarity_search(question)

print(len(docs))

print(docs[0])


from langchain_community.llms import GPT4All

llama3 = GPT4All(model="/home/scifaipy/llms/llama3/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
                 max_tokens=2048,
                 )

response = llama3.invoke("Explain me the explanatory graph paper for CNNs.")

print(response)
