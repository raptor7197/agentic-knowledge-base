from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent , AgentExecutor

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources:list[str]
    tools_used: list[str]



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_template(
    """You are a world-class researcher.
Explain the topic given in {input} in depth.

{format_instructions}

{agent_scratchpad}"""
).partial(format_instructions=parser.get_format_instructions())
# agent = create_tool_calling_agent(llm=llm, prompt=prompt, output_parser=parser)
# agent_executor = AgentExecutor(agent=agent,tools=[],verbose=True)

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=2)
# response = llm.invoke("Explain about Dynamo DB and how it is better than RDS ") used fro directly fetching from the API not through the praser 
# print(response)

# response = agent_executor.invoke({"topic": "Dynamo DB and how it is better than RDS"})
# print(response)

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=[])

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

raw_response = agent_executor.invoke({"input": "DynamoDB and how it is better than RDS"})

try:
    parsed_response = parser.parse(raw_response['output'])
    print(parsed_response)
except Exception as e:
    print("Parsing failed:", e)
    print("Raw response:", raw_response)
