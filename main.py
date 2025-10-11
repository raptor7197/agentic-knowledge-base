from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=2)
response = llm.invoke("Explain about Dynamo DB and how it is better than RDS ")
print(response)