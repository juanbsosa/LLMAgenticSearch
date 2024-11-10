####################################################################################################
########### LLM AGENTIC APP TO RETRIEVE AND EVALUATE WEBPAGES FOR EQUIPMENT PRICES #################
####################################################################################################

#%% LIBRARIES

# For API keys
from dotenv import load_dotenv, find_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# To visualize graph
from IPython.display import Image, display

# Tavily
from tavily import TavilyClient

# Type validation
from typing import TypedDict, List, Dict, Tuple

# To count tokens
import tiktoken


#%% ENVIRONMENT VARIABLES
# Load environment variables: OpenAI, Tavily
# load_dotenv(find_dotenv(), override=True) # !!! Change this when using the correct API key
load_dotenv('/home/sosajuanbautista/aeai-filestore/projects/agentic/codes/price_retrieval/.env', override=True) # !!! Juan's API key. Dont use!



#%% AGENT STATE

# This can also be thought of as the parameters of the function that runs the pipeline.

# Create a class for the agent state
class AgentState(TypedDict):
    commodity: str # Name of commodity class
    search_results_to_evaluate: List[Dict[str, str]]  # Search results to evaluate.
    evaluated_search_results: List[Dict[str, str]] # Place to store the agent's evaluation outcomes for search results.
    min_approved_search_results: int # Minimum number of search results that have to be approved.

    include_justification: bool # Whether to include justification from the evaluation into the refinement node. Default is False.
    use_raw_content: bool # Whether to use the raw content of the search results. If False, uses the condensed version. Default is True.
    max_content_length: int # Maximum character length of the content to be used in the evaluation. Default is 100000.
    min_similarity_score: float # Minimum similarity score (provided by Tavily) for a search result to be evaluated. Default is 0.7.

    n_refined_search_terms: int # Number of refined search terms to create in each iteration.
    max_results_per_search_term: int # Number of results to collect per search term.
    # refined_search_terms: List[Tuple[str, int, int]] # List of new search terms created by the agent, with the corresponding iteration number and model temperature.
    search_terms_temperature_iteration: List[Tuple[str, int, int]] # List of new search terms created by the agent, with the corresponding iteration number and model temperature.
    
    iteration_number: int # Keeps track of the number of iterations (loop: evaluate-refine-search).
    max_iterations: int # Maximum number of iterations allowed.

    n_input_tokens: int # Number of input tokens consumed.
    n_output_tokens: int # Number of output tokens consumed.
    n_tavily_api_calls: int # Number of Tavily API calls made.


#%% STRUCTURED RESPONSES

# Coerce the outcome of the evaluation of a search result to dictionary format
class SearchResultEvaluation(BaseModel):
    evaluation_outcome: bool = Field(description="The outcome of the evaluation of the search result. Equal to True if the search result is relevant to the query, False otherwise.")
    justification: str = Field(description="A brief justification for the evaluation outcome.") # Limited by max_tokens (350)

# Same but without justification
class SearchResultEvaluationShort(BaseModel):
    evaluation_outcome: bool = Field(description="The outcome of the evaluation of the search result. Equal to True if the search result is relevant to the query, False otherwise.")

# Coerce the outcome of the refinement of search terms to be a list of strings
class RefinedSearchTerms(BaseModel):
    refined_search_terms: List[str] = Field(description="The list of refined search terms.")


#%% STATIC AGENTS

# LLM for evaluator node
evaluator_model = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_tokens=350) # !!! Maybe make max_tokens configurable

# The model for refinement will be instantiated within the node to vary temperature dynamically


#%% COUNT TOKENS

# Initialize tokenizer for your model
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(encoding.encode(text))

#%% STATIC PROMPTS

SEARCH_RESULT_EVALUATION_PROMPT_CONTENT = f'''You are an expert analyst tasked with the evaluation of search results for online sources of product prices.

The products are categorized into different categories called "commodities". You will be presented with an individual search result for a particular category of commodities.

Using the information provided in the search result, you have to evaluate if the resulting webpage is useful for the purpose of retrieving prices for products that could be in that commodity category. To guide you in this task, consider the following points:
1. The following common search results should be DISAPPROVED: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations or rankings of the "best" products.
2. The following common search results should be APPROVED: product listings with prices, product catalogs with prices, list of multiple product models with prices.

The search result for the commodity category is presented in dictionary format, where the key "title" is the title of the search result, "url" is the URL of the search result, and "content" is a snippet or brief description of the search result.'''

SEARCH_RESULT_EVALUATION_PROMPT_RAW_CONTENT = f'''You are an expert analyst tasked with the evaluation of search results for online sources of product prices.

The products are categorized into different categories called "commodities". You will be presented with an individual search result for a particular category of commodities.

Using the information provided in the search result, you have to evaluate if the resulting webpage is useful for the purpose of retrieving prices for products that could be in that commodity category. To guide you in this task, consider the following points:
1. The following common search results should be DISAPPROVED: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations or rankings of the "best" products.
2. The following common search results should be APPROVED: product listings with prices, product catalogs with prices, list of multiple product models with prices, online marketplaces, e-commerce platforms.

The search result for the commodity category is presented in dictionary format, where the key "title" is the title of the search result, "url" is the URL of the search result, and "raw_content" is the parsed html content of the webpage.'''


SYSTEM_REFINEMENT_PROMPT = f'''You are an expert analyst tasked with refining search terms to improve the quality of search results for online sources of product prices.

After conducting an initial search, the following criteria were used to evaluate the search results obtained:
1. The following common search results should be DISAPPROVED: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations of the 'best' products, product rankings.
2. The following common search results should be APPROVED: product listings with prices, product catalogs with prices, list of multiple product models with prices, online marketplaces, e-commerce platforms.'''


#%% GRAPH NODES

# EVALUATOR NODE
def evaluate_search_result_node(state: AgentState):

    # Initiate/Get iteration number
    iteration_number = state.get('iteration_number', 0)

    # Load commodity
    commodity = state['commodity']

    # If initial iteration
    if iteration_number==0:
        print('\n')
        print("----------------------------------------------------------")
        print("Performing evaluation for commodity: ", commodity)

        # Reset the evaluated search results
        state['evaluated_search_results'] = []

    # Get the search results evaluated so far
    evaluated_search_results = state.get('evaluated_search_results', [])
    evaluated_urls = [search_result['url'] for search_result in evaluated_search_results]
    if len(evaluated_search_results) > 0:
        print("Amount of unique search results evaluated so far: ", len(evaluated_search_results))

    # Retrieve the search results to evaluate
    search_results_to_evaluate = state['search_results_to_evaluate']

    # Remove the duplicated URLs in the search results to evaluate
    search_results_to_evaluate = remove_duplicates(search_results_to_evaluate)
    print("Amount of search results to evaluate in the current iteration: ", 
          len(search_results_to_evaluate))

    # If initial iteration
    if iteration_number==0:

        # Get the search terms used in the initial search
        initial_search_terms = [search_result['query'] for search_result in search_results_to_evaluate]
        # Only keep unique search terms
        initial_search_terms = list(set(initial_search_terms))

        # Insert initial search terms into refined search terms
        search_terms_temperature_iteration = [(term, 0, None) for term in initial_search_terms]

    else:
        # Retrieve the already existing refined search terms
        search_terms_temperature_iteration = state['search_terms_temperature_iteration']

    # Retrieve input and output tokens
    n_input_tokens = state.get('n_input_tokens', 0)
    n_output_tokens = state.get('n_output_tokens', 0)

    # Evaluate each search result
    for search_result in search_results_to_evaluate:

        # Check if the URL was already evaluated
        if search_result['url'] in evaluated_urls:
            print('Search result already evaluated, ignore it:', search_result['title'], search_result['url'])
            continue

        # Discard search results with low similarity score
        # !!! This should be done in the search node
        if state.get('min_similarity_score'):
            min_similarity_score = state.get('min_similarity_score', 0.7)
            if search_result['score'] < min_similarity_score:
                print(f'Search result discarded due to low similarity score (less than {min_similarity_score}):', search_result['title'], search_result['score'], 
                      search_result['url'])
                continue

        print('Evaluating search result:', search_result['title'], search_result['url'])

        # Prepare the search result content to evaluate
        search_result_content_to_evaluate = {
                'title': search_result['title'],
                'url': search_result['url']
                }
        # Decide whether to use entire raw content or just the content
        # Also limit the length of the content
        use_raw_content = state.get('use_raw_content', True)
        max_content_length = state.get('max_content_length', 100000)
        if use_raw_content:
            if search_result['raw_content'] is not None:
                search_result_content_to_evaluate['raw_content'] = search_result['raw_content']
            else:
                search_result_content_to_evaluate['raw_content'] = search_result['content']
            if max_content_length is not None and search_result_content_to_evaluate['raw_content'] is not None:
                search_result_content_to_evaluate['raw_content'] = search_result_content_to_evaluate['raw_content'][:max_content_length]
        else:
            search_result_content_to_evaluate['content'] = search_result['content']
            if max_content_length is not None and search_result_content_to_evaluate['content'] is not None:
                search_result_content_to_evaluate['content'] = search_result_content_to_evaluate['content'][:max_content_length]

        USER_SEARCH_RESULT_PROMPT = f'''This is the dictionary containing the search result for the commodity category "{commodity}":
        {search_result_content_to_evaluate}.'''
    
        # Messages for the model
        if use_raw_content:
            system_prompt = SEARCH_RESULT_EVALUATION_PROMPT_RAW_CONTENT
        else:
            system_prompt = SEARCH_RESULT_EVALUATION_PROMPT_CONTENT
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=USER_SEARCH_RESULT_PROMPT)
        ]

        # Model with structured output
        include_justification = state.get('include_justification', False)
        if include_justification:
            structured_model = evaluator_model.with_structured_output(SearchResultEvaluation)
        else:
            structured_model = evaluator_model.with_structured_output(SearchResultEvaluationShort)

        # Add input tokens
        for message in messages:
            n_input_tokens += count_tokens(message.content)

        # Invoke model
        response = structured_model.invoke(messages)

        # Add output tokens
        for element in response.dict().values():
            n_output_tokens += count_tokens(str(element))

        # Re-construct evaluated search result
        evaluated_search_result = search_result.copy()
        evaluated_search_result['evaluation_outcome'] = response.evaluation_outcome
        print("Evaluation outcome: ", response.evaluation_outcome)
        evaluated_search_result['justification'] = response.justification

        # Add the evaluated search result to the state with all the evaluated results
        evaluated_search_results.append(evaluated_search_result)

    # Empty the search results to evaluate
    search_results_to_evaluate = []

    # Update the iteration number
    iteration_number += 1

    return {"evaluated_search_results": evaluated_search_results, 
            'search_results_to_evaluate': search_results_to_evaluate,
            'search_terms_temperature_iteration': search_terms_temperature_iteration,
            'iteration_number': iteration_number,
            'n_input_tokens': n_input_tokens,
            'n_output_tokens': n_output_tokens}

# SEARCH-TERM-REFINMENT NODE
def refine_search_term_node(state: AgentState):

    # Initiate a model (the temperature will increase in each iteration)
    iteration_number = state.get('iteration_number')
    temp = round(min(0.1 + 0.2*iteration_number, 1),2)
    print("Temperature set for this iteration: ", temp)
    search_term_refinement_model = ChatOpenAI(model='gpt-4o-mini', temperature=temp)

    # Retrieve the evaluated search results from the state
    evaluated_search_results = state.get('evaluated_search_results', [])
    commodity = state['commodity']
    n_refined_search_terms = state.get('n_refined_search_terms', 1) # default is 1

    # Retrieve the search terms that received approval and disapproval
    all_search_terms = [evaluated_result['query'] for evaluated_result in evaluated_search_results]
    all_search_terms = list(set(all_search_terms))
    approved_search_terms = [evaluated_result['query'] for evaluated_result in evaluated_search_results if evaluated_result['evaluation_outcome']==True]
    approved_search_terms = list(set(approved_search_terms))
    disapproved_search_terms = [evaluated_result['query'] for evaluated_result in evaluated_search_results if evaluated_result['evaluation_outcome']==False]
    disapproved_search_terms = list(set(disapproved_search_terms))

    # Retrieve input and output tokens
    n_input_tokens = state.get('n_input_tokens', 0)
    n_output_tokens = state.get('n_output_tokens', 0)

    # Define the prompt
    USER_REFINEMENT_PROMPT = f'''The end goal is to find sources of price data for the following category of products: {commodity}.
    The initial search was conducted using the following search terms: {all_search_terms}.
    '''

    # Add justification content
    include_justification = state.get('include_justification', False)
    if include_justification:

        # Get some examples of approved search results
        approved_sr1 = None
        approved_sr2 = None

        # Ensure approved_search_terms has at least two elements to access the last two
        if len(approved_search_terms) >= 2:
            # Define the last and second last approved search terms
            last_term = approved_search_terms[-1]
            second_last_term = approved_search_terms[-2]

            # Find the last dictionary matching the last term
            approved_sr1 = next((item for item in reversed(evaluated_search_results) if item['query'] == last_term), None)

            # Find the last dictionary matching the second last term
            approved_sr2 = next((item for item in reversed(evaluated_search_results) if item['query'] == second_last_term), None)

        # If there's only one term in approved_search_terms, check only for the last one
        elif len(approved_search_terms) == 1:
            last_term = approved_search_terms[-1]
            approved_sr1 = next((item for item in reversed(evaluated_search_results) if item['query'] == last_term), None)

        # Get some examples of disapproved search results
        disapproved_sr1 = None
        disapproved_sr2 = None

        # Ensure disapproved_search_terms has at least two elements to access the last two
        if len(disapproved_search_terms) >= 2:
            # Define the last and second last disapproved search terms
            last_term = disapproved_search_terms[-1]
            second_last_term = disapproved_search_terms[-2]

            # Find the last dictionary matching the last term
            disapproved_sr1 = next((item for item in reversed(evaluated_search_results) if item['query'] == last_term), None)

            # Find the last dictionary matching the second last term
            disapproved_sr2 = next((item for item in reversed(evaluated_search_results) if item['query'] == second_last_term), None)

        # If there's only one term in disapproved_search_terms, check only for the last one
        elif len(disapproved_search_terms) == 1:
            last_term = disapproved_search_terms[-1]
            disapproved_sr1 = next((item for item in reversed(evaluated_search_results) if item['query'] == last_term), None)

        # Add them to the prompt
        if disapproved_sr1:
            USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
        These are some of the disapproved search results obtained, including the search term used, the title of the web page, and the justification of why they were disapproved:
            1) Search Term Used: {disapproved_sr1['query']}. Title: {disapproved_sr1['title']}. Justification: {disapproved_sr1['justification']}.'''
        if disapproved_sr2:
            USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
            2) Search Term Used: {disapproved_sr2['query']}. Title: {disapproved_sr2['title']}. Justification: {disapproved_sr2['justification']}.'''

        if approved_sr1:
            USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
            These are some of the approved search results obtained, including the search term used, the title of the web page, and the justification of why they were approved:
            1) Search Term Used: {approved_sr1['query']}. Title: {approved_sr1['title']}. Justification: {approved_sr1['justification']}.'''
        if approved_sr2:
            USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
            2) Search Term Used: {approved_sr2['query']}. Title: {approved_sr2['title']}. Justification: {approved_sr2['justification']}.'''
    
    # Final part of prompt
    USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
    Based on the previous information, provide {n_refined_search_terms} new refined search terms that would yield more relevant results. Do NOT repeat any of the search terms used so far.'''

    # Messages for the model
    messages = [
        SystemMessage(content=SYSTEM_REFINEMENT_PROMPT),
        HumanMessage(content=USER_REFINEMENT_PROMPT)
    ]

    # Model with structured output
    structured_model = search_term_refinement_model.with_structured_output(RefinedSearchTerms)

    # Add input tokens
    for message in messages:
        n_input_tokens += count_tokens(message.content)

    # Invoke model
    response = structured_model.invoke(messages)

    # Add output tokens
    for element in response.dict().values():
        n_output_tokens += count_tokens(str(element))

    # Make sure that the refined search terms have not been used before
    refined_search_terms = response.refined_search_terms
    refined_search_terms = [term for term in refined_search_terms if term not in all_search_terms]

    # Print the refined search terms that were discarded
    if len(response.refined_search_terms) != len(refined_search_terms):
        print("The following refined search terms were discarded because they were already used: ", 
              [term for term in response.refined_search_terms if term in all_search_terms])

    print("Refined search terms: ", refined_search_terms)

    # Add the refined search terms to the state
    search_terms_temperature_iteration = state.get('search_terms_temperature_iteration')
    search_terms_temperature_iteration = search_terms_temperature_iteration + [(term, iteration_number, temp) for term in refined_search_terms]
    
    return {"search_terms_temperature_iteration": search_terms_temperature_iteration,
            'n_input_tokens': n_input_tokens,
            'n_output_tokens': n_output_tokens}


# SEARCH NODE
def search_node(state: AgentState):

    # Retrieve from the state
    search_terms_temperature_iteration = state.get('search_terms_temperature_iteration')

    # Get the search terms for the current iteration
    refined_search_terms = [term for term, iteration, temp in search_terms_temperature_iteration if iteration==state['iteration_number']]

    # Retrieve max_results_per_search_term
    max_results_per_search_term = state.get('max_results_per_search_term', 3) # default is 3
    
    # Create empty list of dictionaries to store search results
    search_results = []

    # Initiate the Tavily client
    from tavily import TavilyClient
    tavily_client = TavilyClient() # API key is stored in the environment variable

    # Get the number of API calls made so far
    n_tavily_api_calls = state.get('n_tavily_api_calls', 0)

    # Make an API call to Tavily
    for term in refined_search_terms:
        print('Searching for:', term)
        response = tavily_client.search(
                query=term, 
                max_results=max_results_per_search_term,
                include_raw_content=True
                )
        n_tavily_api_calls += 1
        print("Amount of results found for this search term: ", len(response['results']))
        for result in response['results']:
            result['query'] = term
            search_results.append(result)

    print("Amount of search results added for evaluation: ", len(search_results))
    # print('AUX. These are the search results to evaluate:', search_results)

    return {"search_results_to_evaluate": search_results,
            'n_tavily_api_calls': n_tavily_api_calls}


#%% CONDITIONAL GRAPH EDGES

# Check if minimum number of approved search results is reached
def count_approved_search_results(state):

    # Get relevant elements from state
    evaluated_search_results = state.get('evaluated_search_results', [])
    min_approved_search_results = state.get('min_approved_search_results', 1)
    iteration_number = state.get('iteration_number')
    print("Just finished iteration number: ", iteration_number-1)
    max_iterations = state.get('max_iterations', 2)

    approved_search_results = [evaluated_result for evaluated_result in evaluated_search_results if evaluated_result['evaluation_outcome']==True]
    print(f"Count of approved search results: {len(approved_search_results)}")

    # Decide whether to continue or end the pipeline
    if len(approved_search_results) >= min_approved_search_results:
        print(f"Minimum number of approved search results ({min_approved_search_results}) reached. End of pipeline.")
        print("Final approved search results: ")
        for approved_result in approved_search_results:
            print(approved_result['title'], approved_result['url'])
        return END
    
    elif (iteration_number-1) > max_iterations:
        print(f"Maximum number of iterations ({max_iterations}) reached. End of pipeline.")
        print("Final approved search results: ")
        for approved_result in approved_search_results:
            print(approved_result['title'], approved_result['url'])
        return END
    
    else:
        print(f"Minimum number of approved search results ({min_approved_search_results}) not reached. Continue to next iteration.")
        return 'refine_search_terms'


#%% DEFINE THE GRAPH

# from langgraph.graph import StateGraph, END

# # Instantiating the state graph builder
# builder = StateGraph(AgentState)

# # Adding nodes to the graph (no particular order needed here, I think)
# # Here we associate the names of the nodes with the functions we defined earlier.
# builder.add_node('evaluator_of_search_results', evaluate_search_result_node)
# builder.add_node('refine_search_terms', refine_search_term_node)
# builder.add_node('search_web', search_node)

# # Setting the entry point of the state graph
# builder.set_entry_point('evaluator_of_search_results')

# # Adding regular edges
# builder.add_edge('refine_search_terms', 'search_web')
# builder.add_edge('search_web', 'evaluator_of_search_results')

# # Adding conditional edges
# builder.add_conditional_edges(
#     'evaluator_of_search_results', # after this
#     count_approved_search_results, # run this function
#     {END: END, 'refine_search_terms': 'refine_search_terms'} # !!! Fix this to expand the graph
# )

# # Finish point
# builder.set_finish_point('evaluator_of_search_results')

#%% COMPILE THE GRAPH

# memory = SqliteSaver.from_conn_string(':memory:')
# graph = builder.compile(checkpointer=MemorySaver())

def initialize_graph(checkpoint_type="memory"):
    """
    Initializes the LangGraph for the LLM agent.
    Parameters:
        checkpoint_type (str): Type of checkpoint storage, "memory" for in-memory, "sqlite" for SQLite.
    Returns:
        graph: Compiled LangGraph instance.
    """

    # Instantiating the state graph builder
    builder = StateGraph(AgentState)

    # Adding nodes to the graph (no particular order needed here, I think)
    # Here we associate the names of the nodes with the functions we defined earlier.
    builder.add_node('evaluator_of_search_results', evaluate_search_result_node)
    builder.add_node('refine_search_terms', refine_search_term_node)
    builder.add_node('search_web', search_node)

    # Setting the entry point of the state graph
    builder.set_entry_point('evaluator_of_search_results')

    # Adding regular edges
    builder.add_edge('refine_search_terms', 'search_web')
    builder.add_edge('search_web', 'evaluator_of_search_results')

    # Adding conditional edges
    builder.add_conditional_edges(
        'evaluator_of_search_results', # after this
        count_approved_search_results, # run this function
        {END: END, 'refine_search_terms': 'refine_search_terms'}
    )

    # Initialize the checkpointer based on the desired checkpoint type
    if checkpoint_type == "sqlite":
        checkpointer = SqliteSaver.from_conn_string(':memory:')
    else:
        checkpointer = MemorySaver()

    # Compile the graph with the checkpointer and return
    return builder.compile(checkpointer=checkpointer)

# #%% OPTION TO RETURN GRAPH
# def get_graph():
#     return graph


#%% VISUALIZE THE GRAPH
def visualize_graph(graph):
    display(Image(graph.get_graph().draw_mermaid_png()))
    

#%% CALCULATE COSTS

# Tavily
def calculate_tavily_search_cost(n_api_calls):
    if n_api_calls<1000:
        return 0
    elif n_api_calls<4000:
        return 30
    elif n_api_calls<15000:
        return 100
    elif n_api_calls<38000:
        return 220
    elif n_api_calls<100000:
        return 500
    else:
        return 500 + ((n_api_calls - 100000)/8000)*100

# OpenAI
def calculate_token_costs(model_name, input_tokens, output_tokens):
    # Define the latest token prices (update these when pricing changes)
    PRICES_PER_1M_TOKENS = {
        "gpt-4o": {"input": 2.5, "output": 10},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6}
    }
    if model_name not in PRICES_PER_1M_TOKENS:
        raise ValueError(f"Model '{model_name}' is not recognized. Update pricing data if needed.")

    price_info = PRICES_PER_1M_TOKENS[model_name]

    input_cost = (input_tokens / 1000000) * price_info["input"]
    output_cost = (output_tokens / 1000000) * price_info["output"]
    
    return input_cost, output_cost


def calculate_costs(
        search_results_to_evaluate, commodity, 
        min_similarity_score=None, use_raw_content=True, include_justification=True,
        justification_length=300, n_refined_search_terms=3, max_iterations=3, max_content_length=100000,
        refined_search_term_length=60, max_results_per_search_term=3, verbose=True,
        price_1m_input_tokens=0.15, price_1m_output_tokens=0.6
                    ):
    
    # TOKENS (count characters and divide by 4 at the end)
    input_tokens_evaluation_node = 0
    output_tokens_evaluation_node = 0
    input_tokens_refinement_node = 0
    output_tokens_refinement_node = 0

    # Search result evaluation

    # Remove the duplicated URLs in the search results to evaluate
    def remove_duplicates(dict_list):
        seen_urls = set()
        unique_dicts = []
        for d in dict_list:
            url = d['url']
            if url not in seen_urls:
                unique_dicts.append(d)  # Add the dictionary if the URL hasn't been seen
                seen_urls.add(url)  # Mark the URL as seen
        return unique_dicts
    search_results_to_evaluate = remove_duplicates(search_results_to_evaluate)

    for search_result in search_results_to_evaluate:

        # Discard search results with low similarity score
        if min_similarity_score:
            if search_result['score'] < min_similarity_score:
                continue

        # Prepare the search result content to evaluate
        search_result_content_to_evaluate = {
                'title': search_result['title'],
                'url': search_result['url']
                }
        if use_raw_content:
            search_result_content_to_evaluate['raw_content'] = search_result['raw_content']
            if max_content_length is not None:
                search_result_content_to_evaluate['raw_content'] = search_result_content_to_evaluate['raw_content'][:max_content_length]
        else:
            search_result_content_to_evaluate['content'] = search_result['content']
            if max_content_length is not None:
                search_result_content_to_evaluate['content'] = search_result_content_to_evaluate['content'][:max_content_length]

        # Add the length of characters of the search result evaluation prompt
        input_tokens_evaluation_node += len(SEARCH_RESULT_EVALUATION_PROMPT_RAW_CONTENT)

        # Add the length of characters of the user search result evaluation prompt
        USER_SEARCH_RESULT_PROMPT = f'''This is the dictionary containing the search result for the commodity category "{commodity}":
            {search_result_content_to_evaluate}.'''
        input_tokens_evaluation_node += len(USER_SEARCH_RESULT_PROMPT)

        # Output tokens
        # False or True
        output_tokens_evaluation_node += 5

        # Justification
        if include_justification:
            output_tokens_evaluation_node += justification_length

    # Estimate additional tokens for the iterations
    input_tokens_evaluation_node += (max_iterations-1)*(n_refined_search_terms*max_results_per_search_term)*len(SEARCH_RESULT_EVALUATION_PROMPT_RAW_CONTENT)
    input_tokens_evaluation_node += (max_iterations-1)*(n_refined_search_terms*max_results_per_search_term)*len(USER_SEARCH_RESULT_PROMPT) # Use the last 'search_result_content_to_evaluate'
    output_tokens_evaluation_node += (max_iterations-1)*(n_refined_search_terms*max_results_per_search_term)*5
    if include_justification:
        output_tokens_evaluation_node += (max_iterations-1)*(n_refined_search_terms*max_results_per_search_term)*justification_length

    # SEARCH-TERM-REFINMENT NODE

    # Retrieve the search terms that received approval and disapproval
    all_search_terms = [evaluated_result['query'] for evaluated_result in search_results_to_evaluate]
    all_search_terms = list(set(all_search_terms))

    # Define the prompt
    USER_REFINEMENT_PROMPT = f'''The end goal is to find sources of price data for the following category of products: {commodity}.
    An initial search was conducted using the following search terms: {all_search_terms}.
    '''

    # Add justification content
    if include_justification:

        # Get the serarch result with the largest title
        search_results_to_evaluate = sorted(search_results_to_evaluate, key=lambda x: len(x['title']), reverse=True)

        USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
        These are some of the disapproved search results obtained, including the search term used, the title of the web page, and the justification of why they were disapproved:
        1) Search Term Used: {search_results_to_evaluate[0]['query']}. Title: {search_results_to_evaluate[0]['title']}. Justification: {'Hello'*justification_length}.
        2) Search Term Used: {search_results_to_evaluate[0]['query']}. Title: {search_results_to_evaluate[0]['title']}. Justification: {'Hello'*justification_length}.

        These are some of the approved search results obtained, including the search term used, the title of the web page, and the justification of why they were approved:
        1) Search Term Used: {search_results_to_evaluate[0]['query']}. Title: {search_results_to_evaluate[0]['title']}. Justification: {'Hello'*justification_length}.
        2) Search Term Used: {search_results_to_evaluate[0]['query']}. Title: {search_results_to_evaluate[0]['title']}. Justification: {'Hello'*justification_length}.
        '''
    
    USER_REFINEMENT_PROMPT = USER_REFINEMENT_PROMPT + f'''
    Based on the previous information, provide {n_refined_search_terms} new refined search terms that would yield more relevant results. Do NOT repeat any of the search terms used so far.'''

    for iteration in range(max_iterations):

        # Input tokens
        input_tokens_refinement_node += len(SYSTEM_REFINEMENT_PROMPT)

        # User input
        input_tokens_refinement_node += len(USER_REFINEMENT_PROMPT)
        # In each new iteration, n_refined_search_terms additional search terms could be added to all_search_terms
        # In the first iteration, there is no need to add anything else
        input_tokens_refinement_node += refined_search_term_length*iteration

        # Output tokens
        output_tokens_refinement_node += n_refined_search_terms*refined_search_term_length

    # TAVILY SEARCH NODE
    n_tavily_api_calls = 0
    n_tavily_api_calls += max_iterations-1

    # Divide characters by 4 to get tokens
    input_tokens_evaluation_node = round(input_tokens_evaluation_node / 4, 0)
    output_tokens_evaluation_node = round(output_tokens_evaluation_node / 4, 0)
    input_tokens_refinement_node = round(input_tokens_refinement_node / 4, 0)
    output_tokens_refinement_node = round(output_tokens_refinement_node / 4, 0)

    total_input_tokens = input_tokens_evaluation_node + input_tokens_refinement_node
    total_output_tokens = output_tokens_evaluation_node + output_tokens_refinement_node

    total_cost_input_tokens, total_cost_output_tokens = calculate_token_costs("gpt-4o-mini", total_input_tokens, total_output_tokens)

    total_cost_tokens = total_cost_input_tokens + total_cost_output_tokens

    tavily_monthly_cost = calculate_tavily_search_cost(n_tavily_api_calls)

    if verbose:
        print("Total input tokens in evaluation node: ", input_tokens_evaluation_node)
        print("Total output tokens in evaluation node: ", output_tokens_evaluation_node)
        print("Total input tokens in refinement node: ", input_tokens_refinement_node)
        print("Total output tokens in refinement node: ", output_tokens_refinement_node)

        print("Total input tokens: ", total_input_tokens)
        print("Total cost of input tokens: ", total_cost_input_tokens)
        print("Total output tokens: ", total_output_tokens)
        print("Total cost of output tokens: ", total_cost_output_tokens)

        print("Total token cost: ", total_cost_tokens)

        print("Total number of Tavily API calls: ", n_tavily_api_calls)
        print("Total monthly cost of Tavily searches: ", tavily_monthly_cost)

    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_token_cost': total_cost_tokens,
        'n_tavily_api_calls': n_tavily_api_calls
        # 'tavily_monthly_cost': tavily_monthly_cost
    }

#%% UTILS

# Remove the duplicated URLs in the search results to evaluate
def remove_duplicates(dict_list):
    seen_urls = set()
    unique_dicts = []
    for d in dict_list:
        url = d['url']
        if url not in seen_urls:
            unique_dicts.append(d)  # Add the dictionary if the URL hasn't been seen
            seen_urls.add(url)  # Mark the URL as seen
    return unique_dicts


# #%% 

# if __name__ == "__main__":

#     graph = initialize_graph("memory")  
#     # Default settings for running the graph when the script is executed directly
#     thread = {'configurable': {'thread_id': '1'}}
#     prompt = {
#         'commodity': 'Rotary tiller or power tiller',
#         'search_results_to_evaluate': [],  # Replace with actual default value
#         'min_approved_search_results': 10,
#         'n_refined_search_terms': 3,
#         'max_results_per_search_term': 2,
#         'max_iterations': 3,
#         'include_justification': False,
#         'use_raw_content': True
#     }

#     # Run graph and get response
#     response = graph.invoke(prompt, thread)
#     print(response)
