from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = CSVLoader(
    file_path="realistic_restaurant_reviews.csv",
    encoding="utf-8"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

model = OllamaLLM(model="llama3.2")

template = """ You are an expert in answering question about a pizza restaurant. Talk like a super MARIO Bros character.here are some relevant reviews:{reviews}
Question: {question}"""

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(
    collection_name="pizza_reviews",
    embedding_function=embeddings,
)
db.add_documents(chunks)
retriever = db.as_retriever(search_kwargs={"k": 5})
prompt = ChatPromptTemplate.from_template(template)
chain = prompt|model

reviews = retriever.invoke("What is the best pizza place in town?")
result = chain.invoke({"reviews":[reviews],"question":"What is the best pizza place in town?"})
print(result)
