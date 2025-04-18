import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

from crewai import Agent, Task, Crew
from crewai import LLM

llm = LLM(
    model="openai/gpt-4", # call model by provider/model_name
    temperature=0.8,
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42
)

# Define your agents with roles and goals
researcher = Agent(
    role='Researcher',
    goal='Discover and analyze cutting-edge developments in AI technology',
    backstory='You are an expert technology researcher with deep knowledge of artificial intelligence trends',
    verbose=True,
    llm=llm
)

writer = Agent(
    role='Technical Writer',
    goal='Create clear and engaging content explaining complex AI concepts',
    backstory='You are a skilled technical writer who specializes in making complex topics accessible',
    verbose=True,
    llm=llm
)

# Create tasks for your agents
research_task = Task(
    description='Research the latest developments in large language models',
    expected_output='A comprehensive summary of recent advances in LLMs with key insights',
    agent=researcher
)

analysis_task = Task(
    description='Analyze potential business applications of recent AI advancements',
    expected_output='A detailed report on how new AI technologies can be applied in various industries',
    agent=researcher
)

writing_task = Task(
    description='Write a blog post explaining how large language models work to a general audience',
    expected_output='A well-structured, engaging blog post of 500 words that explains LLMs in simple terms',
    agent=writer
)

# Create a crew with your agents and tasks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, analysis_task, writing_task],
    verbose=True
)

# Start the crew
result = crew.kickoff()

print("\n==== CREW RESULT ====\n")
print(result)
