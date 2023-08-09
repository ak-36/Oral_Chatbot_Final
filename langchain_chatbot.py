import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# key = os.environ['OPENAI_API_KEY']

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import  RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import OpenAI
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_File_Object = file
    # Here, we will creat a pdf reader object
        pdf_Reader = PyPDF2.PdfReader(pdf_File_Object)
    # Now we will print number of pages in pdf file
        print("No. of pages in the given PDF file: ", len(pdf_Reader.pages))
        num_pages = len(pdf_Reader.pages)
        for j in range(num_pages):
    # Here, create a page object
          page_Object = pdf_Reader.pages[j]
    # Now, we will extract text from page
          text = text + page_Object.extract_text()
    # At last, close the pdf file object
        pdf_File_Object.close()
    return text

def load_multiple_pdfs(folder_path):
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_texts[filename] = extract_text_from_pdf(pdf_path)
    return pdf_texts

def splitter(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    return text_splitter.split_text(raw_text)

def embedder(texts):
# Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def model(docsearch):
    chat = OpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})
    chain = RetrievalQA.from_llm(llm = chat, retriever=retriever)
    return chain

    
def prepare_context(folder):
    pdf_texts = load_multiple_pdfs(folder)
    print("Files loaded")
    raw_text=""
    for i in pdf_texts.keys():
        raw_text = raw_text + pdf_texts[i]
    texts = splitter(raw_text)
    print("Splitting of Data completed")
    docsearch = embedder(texts)
    print("Embedding of Data completed")
    chain = model(docsearch)
    print("Chain created")
    return chain

def get_response(user_input, chain):
    if user_input.lower() in ["exit", "quit", "bye"]:
        response ={'result' : "Goodbye!"}  # Exit the loop if the user wants to end the conversation
    else:
        response = chain(user_input)
    print("Answer", response)
    return response
