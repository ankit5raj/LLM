from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


#promp1
template1 = PromptTemplate(
    template= 'write a detailed report on {topic}',
    input_variables= ['topic']
)


#promp2
template2 = PromptTemplate(
    template= 'write a 5 line summary on the following text. /n {text}',
    input_variables= ['text']
)

prompt1 = template1.invoke({'topic':'M.S Dhoni'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

final = model.invoke(prompt2)

print(final.content)