import os
import logging
from typing import List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables and configure API keys."""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'LANGCHAIN_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Set environment variables for API keys
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
    
    logger.info("Environment variables loaded successfully")

def initialize_llm() -> LLM:
    """Initialize and configure the language model."""
    return LLM(
        model="openai/gpt-4",
        temperature=0.7,  # Slightly reduced for more focused outputs
        max_tokens=200,   # Increased for more detailed responses
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=["END"],
        seed=42
    )
serper_tool = SerperDevTool(name="serper_tool", description="A tool for searching the web for the latest news and information",n=2)

def create_agents(llm: LLM) -> List[Agent]:
    """Create and return a list of agents with defined roles."""
    researcher = Agent(
        role='Researcher',
        goal='Discover and analyze cutting-edge developments in AI technology',
        backstory='You are an expert technology researcher with deep knowledge of artificial intelligence trends',
        verbose=True,
        llm=llm,
        tools=[serper_tool]
    )
    
    writer = Agent(
        role='Technical Writer',
        goal='Create clear and engaging content explaining complex AI concepts',
        backstory='You are a skilled technical writer who specializes in making complex topics accessible',
        verbose=True,
        llm=llm,
    )
    
    return [researcher, writer]

def create_tasks(agents: List[Agent]) -> List[Task]:
    """Create and return a list of tasks for the agents."""
    researcher, writer = agents
    
    tasks = [
        Task(
            description='Research the latest developments in large language models',
            expected_output='A comprehensive summary of recent advances in LLMs with key insights',
            agent=researcher
        ),
        Task(
            description='Analyze potential business applications of recent AI advancements',
            expected_output='A detailed report on how new AI technologies can be applied in various industries',
            agent=researcher
        ),
        Task(
            description='Write a blog post explaining how large language models work to a general audience',
            expected_output='A well-structured, engaging blog post of 500 words that explains LLMs in simple terms',
            agent=writer
        )
    ]
    
    return tasks

def main():
    """Main function to orchestrate the AI crew workflow."""
    try:
        # Setup environment
        load_environment()
        
        # Initialize language model
        llm = initialize_llm()
        logger.info("Language model initialized")
        
        # Create agents and tasks
        agents = create_agents(llm)
        tasks = create_tasks(agents)
        logger.info(f"Created {len(agents)} agents and {len(tasks)} tasks")
        
        # Create and run the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )
        
        logger.info("Starting the crew...")
        result = crew.kickoff()
        
        print("\n==== CREW RESULT ====\n")
        print(result)
        return result
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    main()
