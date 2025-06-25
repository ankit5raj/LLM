from langchain_groq import ChatGroq
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

promt1 = PromptTemplate(
    template= ' Generate a detailed report on {topic}',
    input_variables= ['topic']
)
promt2 = PromptTemplate( 
    template= ' Generate a 5 pointer summary from the following text \n {text}',
    input_variables= ['text']
)

parser = StrOutputParser()

chain = promt1 | model | parser | promt2 | model | parser 

result = chain.invoke({'topic':'IITs'})

print(result)

chain.get_graph().print_ascii()