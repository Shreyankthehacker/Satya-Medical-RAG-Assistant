from chains import fallback_chain,grader_chain,rag_chain,answer_grader_chain,hallucination_grader_chain
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

urls = [
    "https://www.webmd.com/a-to-z-guides/malaria",
    "https://www.webmd.com/diabetes/type-1-diabetes",
    "https://www.webmd.com/diabetes/type-2-diabetes",
    "https://www.webmd.com/migraines-headaches/migraines-headaches-migraines",
]


loader = WebBaseLoader(
    urls , 
    bs_get_text_kwargs={'strip':True}
)

docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 30)
chunks = splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(documents = chunks , embedding  = embedding_function)
retriever = vector_store.as_retriever()
router_prompt_temp = (
     "You are an expert in routing user queries to either a VectorStore, SearchEngine\n"
    "Use SearchEngine for all other medical queries that are not related to malaria, diabetes, or migraines.\n"
    "The VectorStore contains information on malaria, diabetes, and migraines.\n"
    'Note that if a query is not medically-related, you must output "not medically-related", don\'t try to use any tool.\n\n'
    "query: {query}"
)




class VectorStore(BaseModel):
    (
        "A vectorstore contains information about symptoms, treatment"
        ", risk factors and other information about malaria, type 1 and"
        "type 2 diabetes and migraines"
    )
    query: str  

class SearchEngine(BaseModel):
    ''' Searhc engine for othe medical info in web for that formatting here'''
    query : str


router_prompt_temp = (
     "You are an expert in routing user queries to either a VectorStore, SearchEngine\n"
    "Use SearchEngine for all other medical queries that are not related to malaria, diabetes, or migraines.\n"
    "The VectorStore contains information on malaria, diabetes, and migraines.\n"
    'Note that if a query is not medically-related, you must output "not medically-related", don\'t try to use any tool.\n\n'
    "query: {query}"
)




llm = ChatGroq(model="llama3-70b-8192", temperature=0)



prompt = ChatPromptTemplate.from_template(router_prompt_temp)

tools = [VectorStore , SearchEngine]

llm_with_tools = llm.bind_tools(tools)

question_router = prompt | llm_with_tools




class AgentState(TypedDict):
    """The dictionary keeps track of the data required by the various nodes in the graph"""
    query : str
    generation : str
    chat_history : list[BaseMessage]
    documents : list[Document]

def retrieve_node(state:AgentState) -> dict[str,list[Document] | str] :
    """
    Retrieve relevent documents from the vectorstore

    query: str

    return list[Document]
    """
    print(f"retrieve node ")

    query = state['query']
    documents = retriever.invoke(input = query)
    return {"documents":documents}

def fallback_node(state:AgentState):
    ''' Fallback to this node when there is no tool call'''
    print(f"fallback node ")
    query = state['query']
    chat_history = state['chat_history']
    generation = fallback_chain.invoke({"query":query,'chat_history':chat_history})
    return {"generation":generation}
def filter_documents_node(state:AgentState):
    filtered_docs = list()

    query = state['query']
    documents = state['documents']
    print(f"filter docs node ")

    for i,docs in enumerate(documents,start = 1):
        grade = grader_chain.invoke({"query":query,"context":docs})
        if grade.grade == 'relevant':
            print(f"Chuck.......{i} is relevent")
            filtered_docs.append(docs)
        else:
            print(f"Chuck.....{i} is irrelevent")
    return {"documents":filtered_docs}             

def rag_node(state:AgentState):
    print(f"rag node ")
    query = state['query']
    documents = state['documents']

    generation = rag_chain.invoke({"query":query , 'context':documents})
    return {"generation": generation}


tavily_search = TavilySearchResults()
def web_search_node(state:AgentState):
    print(f"search node ")
    query = state['query']
    results = tavily_search.invoke(query)
    documents = [
        Document(page_content = doc['content'],metadata = {'source':doc['url']}) for doc in results
        
    ]
    return {"documents":documents}

def question_router_node(state:AgentState):
    print("router node")
    query = state['query']
    try:
        response = question_router.invoke({'query':query})
    except Exception:
        return "llm_feedback"

    if 'tool_calls' not in response.additional_kwargs:
        print('-----No tools called--------')
        return 'llm_feedback'
    if len(response.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide route!"
    
    route = response.additional_kwargs['tool_calls'][0]['function']['name']

    if route =='VectorStore':
        print("Routing to the vector store....")
        return "VectorStore"
    
    elif route == 'SearchEngine':
        print("Routing to search enginee")

        return "SearchEngine"
    


def should_generate(state: dict):
    print("should generate node")
    filtered_docs = state["documents"]

    if not filtered_docs:
        print("---All retrived documents not relevant---")
        return "SearchEngine"
    else:
        print("---Some retrived documents are relevant---")
        return "generate"


def hallucination_and_answer_relevance_check(state: dict):
    print("hallucination node")
    llm_response = state["generation"]
    documents = state["documents"]
    query = state["query"]

    hallucination_grade = hallucination_grader_chain.invoke(
        {"response": llm_response, "context": documents}
    )
    if hallucination_grade.grade == "no":
        print("---Hallucination check passed---")
        answer_relevance_grade = answer_grader_chain.invoke(
            {"response": llm_response, "query": query}
        )
        if answer_relevance_grade.grade == "yes":
            print("---Answer is relevant to question---\n")
            return "useful"
        else:
            print("---Answer is not relevant to question---")
            return "not useful"
    print("---Hallucination check failed---")
    return "generate"