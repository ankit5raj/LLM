from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

result = model.invoke("what is the capital of india")
print(result.content)