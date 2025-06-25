from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

class person(BaseModel):
    name: str = Field(description= 'Name of the person'),
    age: int = Field(description= 'Age of the person'),
    city: str = Field(description= 'Name of the city of the person belongs to')


parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template= ' Generate name, age, city of a fictional {place} person \n {format_instruction}',
    input_variables= ['place'],
    partial_variables= {"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'indori'})

print(final_result)