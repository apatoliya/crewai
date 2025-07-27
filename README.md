# CrewAI Project

This project demonstrates the use of CrewAI for orchestrating multiple AI agents to perform research and content creation tasks. It also provides a quick introduction to FastAPI for building modern APIs in Python.

## Features
- Uses CrewAI to coordinate a Researcher and Technical Writer agent
- Agents perform research, analysis, and content writing tasks
- Modular, production-ready Python code with logging and error handling
- Example FastAPI setup instructions for building APIs

## Requirements
- Python 3.7+
- Conda or virtualenv (recommended)
- Required Python packages (see below)

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   conda create -n crewai-env python=3.9 -y
   conda activate crewai-env
   # or using venv
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Create a `.env` file in the project root with the following:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     LANGCHAIN_API_KEY=your_langchain_api_key
     ```

## Running the CrewAI Script

```sh
python app.py
```

This will run the agents and print the results to the console.

