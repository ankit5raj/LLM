from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

parser = JsonOutputParser()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)


#promp1
template = PromptTemplate(
    template= " give me the indian name, age and city of a fictional person \n {format_instruction}",
    input_variables= [],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()

# result  = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser
result = chain.invoke({})

print(result)