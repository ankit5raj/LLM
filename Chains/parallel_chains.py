from langchain_groq import ChatGroq
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

model2 = ChatGroq(
    model="deepseek-r1-distill-llama-70b"
)

prompt1 = PromptTemplate(
    template= 'generate short and simple notes from the following prompt \n {text}',
    input_variables= ['text']
)

prompt2 = PromptTemplate(
    template= 'generate 5 short question answers from the following text \n {text}',
    input_variables= ['text']
)

prompt3 = PromptTemplate(
    template= ' Merge the provide notes and quiz into a single document \n notes -> {notes} and quiz-> {quiz}',
    input_variables= ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain =RunnableParallel( {
    'notes': prompt1| model | parser,
    'quiz': prompt2 | model2 |parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain 

text = """
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. It aims to find the "best fit" line (or hyperplane in higher dimensions) that represents the relationship between the variables. 
Key Concepts:
Dependent Variable:
The variable being predicted or explained (often denoted as 'y'). 
Independent Variable(s):
The variable(s) used to predict the dependent variable (often denoted as 'x'). 
Linear Equation:
The equation of a line (or hyperplane) that represents the relationship, e.g., y = mx + b for simple linear regression. 
Coefficients:
The values 'm' (slope) and 'b' (intercept) in the linear equation that define the line's position and orientation. 
Residuals:
The differences between the actual values and the predicted values by the model. Minimizing the sum of squared residuals is a common goal in linear regression. 
Types of Linear Regression: 
Simple Linear Regression: Involves one independent variable and one dependent variable. 
Multiple Linear Regression: Involves two or more independent variables and one dependent variable. 
Applications:
Prediction:
Estimating the value of a dependent variable based on the values of independent variables. 
Understanding Relationships:
Identifying the strength and direction of the relationship between variables. 
Business and Finance:
Analyzing sales data, predicting stock prices, and understanding customer behavior. 
Machine Learning:
Linear regression is a fundamental supervised learning algorithm used for classification and regression tasks. 
Interpreting Results:
Slope (m):
Represents the change in the dependent variable for each unit change in the independent variable.
Intercept (b):
Represents the predicted value of the dependent variable when the independent variable is zero. 
Example:
Imagine you want to predict a house's selling price based on its size. You could use linear regression. If the regression equation is Price = 1000 * Size + 50000, it suggests that for every additional square foot (increase in size), the price is predicted to increase by $1000, and a house with 0 square feet would have a predicted price of $50000 (which is not realistic in this context, but demonstrates the intercept). 
"""

result = chain.invoke({'text':text})

print(result)
