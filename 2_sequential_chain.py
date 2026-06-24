from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

os.environ['LANGSMITH_PROJECT']='Sequential__chain_project'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)


model1= ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.8,
)

model2= ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.5,
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config={"run_name":'sequential llm app','tags':['llm app','report generation']}

result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
