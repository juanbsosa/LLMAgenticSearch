# LLM Agentic Search Application (Work in Progress)

## 🚧 Ongoing Development 🚧

**Note**: This repository is a work in progress, and further changes are expected soon. The current implementation serves as an initial prototype and is part of a broader project in which I am involved, aimed at building advanced tools for dynamic data retrieval using Language Models (LLMs) (https://www.machineryofprogress.com/, led by Peter Lambert and Yannick Schindler).

## Overview

This project features a Python application that employs an agentic Language Model (LLM) to search, retrieve, and evaluate pricing information for various commodities from online sources. The application integrates several state-of-the-art libraries, including **LangChain**, **LangGraph**, and **Tavily**, to facilitate a robust and adaptive querying process.

### Key Components

- **llm_agentic_app_for_price_sources.py**: This is the core script of the project, defining the agentic LLM application. It manages the agent's state, processes search queries, and evaluates the relevance of web content for extracting pricing data.
- **build_raw_response_analysis.ipynb**: A Jupyter Notebook designed to analyze the raw responses obtained from the agentic application. It includes tools for inspecting the LLM output and refining the data analysis process.
- **run_retrieve_price_sources.ipynb**: An entry-point Jupyter Notebook for executing the LLM-powered search and retrieval process. It initializes the environment and guides the user through the data retrieval pipeline.

### Features

- **Adaptive State Management**: Uses `TypedDict` data structures to track and manage the agent's state, allowing the application to handle complex, dynamic search queries.
- **Web Integration**: Incorporates **Tavily** for executing web searches and extracting relevant data directly from online sources.
- **Token Optimization**: Utilizes **tiktoken** for efficient token counting, helping to minimize API usage costs when interacting with OpenAI's language models.
- **Modular and Extensible Architecture**: Leverages **LangChain** and **LangGraph** for building a flexible application framework that can be easily extended or modified.

## Prerequisites

- Python 3.9 or higher
- Jupyter Notebook
- API keys for:
  - OpenAI
  - Tavily

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LLMAgenticSearch-main
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment variables:
- Create a `.env` file with your API keys. For example:
   ```bash
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   ```

## Usage

### Running the LLM Agentic Application
1. Open the Jupyter Notebook run_retrieve_price_sources.ipynb.
2. Follow the steps in the notebook to initialize the environment and execute the retrieval process.

### Analyzing Raw Responses
1. Open the Jupyter Notebook build_raw_response_analysis.ipynb.
2. Execute the notebook cells to load and analyze the raw data generated by the LLM agent.
