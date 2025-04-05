
from pydantic import BaseModel,field_validator, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os 
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import HumanMessage
from operator import itemgetter


GROQ_API_KEY = os.getenv("GROQQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]="advanced-rag"
os.environ["TAVILY_API_KEY"]=TAVILY_API_KEY


llm = ChatGroq(model="llama3-70b-8192", temperature=0)

class Grader(BaseModel):
    """This format checks how relevant the retrieved docs are."""
    
    grade: Literal["relevant", "irrelevant"]

    @field_validator("grade", mode="before")
    def validate_grade(cls, value):
        if value == "not relevant":
            return "irrelevant"
        return value
grader_system_prompt_template = """"You are a grader tasked with assessing the relevance of a given context to a query. 
    If the context is relevant to the query, score it as "relevant". Otherwise, give "irrelevant".
    Do not answer the actual answer, just provide the grade in JSON format with "grade" as the key, without any additional explanation."
    """


grader_prompt = ChatPromptTemplate.from_messages([
    ("system",grader_system_prompt_template),
    ("human","context is : {context}\n\n query : {query}")
])

llm_with_structured = llm.with_structured_output(Grader , method = 'json_mode')

grader_chain = grader_prompt | llm_with_structured

rag_template_str = (
    "You are a helpful assistant. Answer the query below based only on the provided context.\n\n"
    "context: {context}\n\n"
    "query: {query}"
)

rag_prompt = ChatPromptTemplate.from_template(rag_template_str)



rag_chain = rag_prompt | llm | StrOutputParser()


fall_back_template =  "You are a friendly medical assistant created by NHVAI.\n"
"Do not respond to queries that are not related to health.\n"
"If a query is not related to health, acknowledge your limitations.\n"
"Provide concise responses to only medically-related queries.\n\n"
"Current conversations:\n\n{chat_history}\n\n"
"human: {query}"



fall_back_prompt = ChatPromptTemplate.from_template(fall_back_template)




chat_history = lambda x: "\n".join(
            [
                (
                    f"human: {msg.content}"
                    if isinstance(msg, HumanMessage)
                    else f"AI: {msg.content}"
                )
                for msg in x["chat_history"]
            ]
)


fallback_chain  = (
    {"chat_history":chat_history , "query":itemgetter("query")}
    | fall_back_prompt
    | llm 
    | StrOutputParser()
)

class HallucinationGrader(BaseModel):
    "Binary score for hallucination check in llm's response"

    grade: Literal["yes", "no"] = Field(
        ..., description="'yes' if the llm's reponse is hallucinated otherwise 'no'"
    )


hallucination_grader_system_prompt_template = (
    "You are a grader assessing whether a response from an llm is based on a given context.\n"
    "If the llm's response is not based on the given context give a score of 'yes' meaning it's a hallucination"
    "otherwise give 'no'\n"
    "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
)

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system_prompt_template),
        ("human", "context: {context}\n\nllm's response: {response}"),
    ]
)


hallucination_grader_chain = (
    RunnableParallel(
        {
            "response": itemgetter("response"),
            "context": lambda x: "\n\n".join([c.page_content for c in x["context"]]),
        }
    )
    | hallucination_grader_prompt
    | llm.with_structured_output(HallucinationGrader, method="json_mode")
)

class AnswerGrader(BaseModel):
    "Binary score for an answer check based on a query."

    grade: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the provided answer is an actual answer to the query otherwise 'no'",
    )


answer_grader_system_prompt_template = (
    "You are a grader assessing whether a provided answer is in fact an answer to the given query.\n"
    "If the provided answer does not answer the query give a score of 'no' otherwise give 'yes'\n"
    "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
)

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system_prompt_template),
        ("human", "query: {query}\n\nanswer: {response}"),
    ]
)


answer_grader_chain = answer_grader_prompt | llm.with_structured_output(
    AnswerGrader, method="json_mode"
)


router_prompt_temp = (
     "You are an expert in routing user queries to either a VectorStore, SearchEngine\n"
    "Use SearchEngine for all other medical queries that are not related to malaria, diabetes, or migraines.\n"
    "The VectorStore contains information on malaria, diabetes, and migraines.\n"
    'Note that if a query is not medically-related, you must output "not medically-related", don\'t try to use any tool.\n\n'
    "query: {query}"
)





