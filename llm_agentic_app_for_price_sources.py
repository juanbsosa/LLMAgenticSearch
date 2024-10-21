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
from typing import TypedDict, List, Dict


#%% ENVIRONMENT VARIABLES
# Load environment variables: OpenAI, Tavily
# load_dotenv(find_dotenv(), override=True) # !!! Change this when using the correct API key
load_dotenv('/home/sosajuanbautista/aeai-filestore/projects/agentic/codes/price_retrieval/.env', override=True) # !!! Juan's API key. Dont use!



#%% AGENT STATE

# This can also be thought of as the parameters of the function that runs the pipeline.

# Create a class for the agent state
class AgentState(TypedDict):
    commodity: str # Name of commodity class
    search_results_to_evaluate: List[Dict[str, str]]  # List of search results to evaluate.
    evaluated_search_results: List[Dict[str, str]] # Dictionary with the agent's evaluation (and justification).
    min_approved_search_results: int # Minimum number of search results that need to be approved.
    iteration_number: int # Keeps track of the number of iterations made to
    max_iterations: int # Maximum number of iterations allowed.
    n_refined_search_terms: int # Number of refined search terms.
    max_results_per_search_term: int # Number of results per search term.
    refined_search_terms: List[str] # List of refined search terms.
    include_justification: bool # Whether to include justification in the evaluation. Default is False
    use_raw_content: bool # Whether to use the raw content of the search results. If False, uses the condense version. Default is True
    min_similarity_score: float # Minimum similarity score for search results to be considered. Default is 0.7


#%% STRUCTURED RESPONSES

# Coerce the outcome of the evaluation of a search result to dictionary format
class SearchResultEvaluation(BaseModel):
    evaluation_outcome: bool = Field(description="The outcome of the evaluation of the search result. Equal to True if the search result is relevant to the query, False otherwise.")
    justification: str = Field(description="A brief justification for the evaluation outcome.")

# Same but without justification
class SearchResultEvaluationShort(BaseModel):
    evaluation_outcome: bool = Field(description="The outcome of the evaluation of the search result. Equal to True if the search result is relevant to the query, False otherwise.")

# Coerce the outcome of the refinement of search terms to be a list of strings
class RefinedSearchTerms(BaseModel):
    refined_search_terms: List[str] = Field(description="The list of refined search terms.")


#%% STATIC AGENTS

# LLM for evaluator node
evaluator_model = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_tokens=350)

# The model for refinement will be instantiated within the node to vary temperature dynamically

#%% STATIC PROMPTS

# Prompt for evaluation of serach results (fixed for all commodities)
# SEARCH_RESULT_EVALUATION_PROMPT = f'''You are an expert analyst tasked with the evaluation of search results for online sources of product prices.

# The products are categorized into different categories called "commodities". You will be presented with an individual search result for a particular category of commodities.

# Using the information provided in the search result, you have to evaluate if the resulting webpage is useful for the purpose of retrieving prices for products that could be in that commodity category. To guide you in this task, consider the following points:
# 1. The following common search results are NOT USEFUL for the task: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations or rankings of the "best" products.
# 2. The following common search results are USEFUL for the task: product listings with prices, product catalogs with prices, list of multiple product models with prices.

# The search result for the commodity category is presented in dictionary format, where the key "query" is the search term used to obtain the search result, "title" is the title of the search result, "url" is the URL of the search result, "content" is a snippet or brief description of the search result, "score" is a similarity score outputed by the search engine, and "raw_content" is the parsed html content of the webpage.'''

SEARCH_RESULT_EVALUATION_PROMPT_CONTENT = f'''You are an expert analyst tasked with the evaluation of search results for online sources of product prices.

The products are categorized into different categories called "commodities". You will be presented with an individual search result for a particular category of commodities.

Using the information provided in the search result, you have to evaluate if the resulting webpage is useful for the purpose of retrieving prices for products that could be in that commodity category. To guide you in this task, consider the following points:
1. The following common search results are NOT USEFUL for the task: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations or rankings of the "best" products.
2. The following common search results are USEFUL for the task: product listings with prices, product catalogs with prices, list of multiple product models with prices.

The search result for the commodity category is presented in dictionary format, where the key "title" is the title of the search result, "url" is the URL of the search result, and "content" is a snippet or brief description of the search result.'''

SEARCH_RESULT_EVALUATION_PROMPT_RAW_CONTENT = f'''You are an expert analyst tasked with the evaluation of search results for online sources of product prices.

The products are categorized into different categories called "commodities". You will be presented with an individual search result for a particular category of commodities.

Using the information provided in the search result, you have to evaluate if the resulting webpage is useful for the purpose of retrieving prices for products that could be in that commodity category. To guide you in this task, consider the following points:
1. The following common search results are NOT USEFUL for the task: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations or rankings of the "best" products.
2. The following common search results are USEFUL for the task: product listings with prices, product catalogs with prices, list of multiple product models with prices.

The search result for the commodity category is presented in dictionary format, where the key "title" is the title of the search result, "url" is the URL of the search result, and "raw_content" is the parsed html content of the webpage.'''


#%% GRAPH NODES

# EVALUATOR NODE
def evaluate_search_result_node(state: AgentState):

    # Initiate/Get iteration number
    iteration_number = state.get('iteration_number', 1)

    # Load commodity
    commodity = state['commodity']
    if iteration_number==1:
        print("Performing evaluation for commodity: ", commodity)

    # Get the search results evaluated so far
    evaluated_search_results = state.get('evaluated_search_results', [])
    evaluated_urls = [search_result['url'] for search_result in evaluated_search_results]
    if len(evaluated_search_results) > 0:
        print("Amount of search results evaluated so far: ", len(evaluated_search_results))

    # Retrieve the evaluated search results from the state
    search_results_to_evaluate = state['search_results_to_evaluate']

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
    print("Amount of search results to evaluate: ", len(search_results_to_evaluate))

    for search_result in search_results_to_evaluate:

        # First check if the search result is already evaluated
        if search_result['url'] in evaluated_urls:
            print('Search result already evaluated, ignore:', search_result['title'], search_result['url'])
            continue

        print('Evaluating search result:', search_result['title'], search_result['url'])

        # Discard search results with low similarity score
        if state.get('min_similarity_score'):
            min_similarity_score = state.get('min_similarity_score', 0.7)
            if search_result['score'] < min_similarity_score:
                print(f'Search result discarded due to low similarity score (less than {min_similarity_score}):', search_result['title'], search_result['url'])
                continue

        # Prepare the search result content to evaluate
        search_result_content_to_evaluate = {
                'title': search_result['title'],
                'url': search_result['url']
                }
        use_raw_content = state.get('use_raw_content', True)
        if use_raw_content:
            search_result_content_to_evaluate['raw_content'] = search_result['raw_content']
        else:
            search_result_content_to_evaluate['content'] = search_result['content']

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

        # Invoke model
        response = structured_model.invoke(messages)

        # Re-construct evaluated search result
        evaluated_search_result = search_result.copy()
        evaluated_search_result['evaluation_outcome'] = response.evaluation_outcome
        print("Evaluation outcome: ", response.evaluation_outcome)
        evaluated_search_result['justification'] = response.justification

        # Add the evaluated search result to the state with all the evaluated results
        evaluated_search_results.append(evaluated_search_result)

    # Empty the search results to evaluate
    state['search_results_to_evaluate'] = []

    # Update the iteration number
    iteration_number = state.get('iteration_number', 0)
    iteration_number += 1

    return {"evaluated_search_results": evaluated_search_results, 
            'search_results_to_evaluate': search_results_to_evaluate,
            'iteration_number': iteration_number}

# SEARCH-TERM-REFINMENT NODE
def refine_search_term_node(state: AgentState):

    # Initiate a model (the temperature will increase in each iteration)
    iteration_number = state.get('iteration_number')
    temp = min(0.1 + 0.2*iteration_number, 1)
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

    # Define the prompt
    SYSTEM_REFINEMENT_PROMPT = f'''You are an expert analyst tasked with refining search terms to improve the quality of search results for online sources of product prices for the following category of products: {commodity}.

    An initial search was conducted using the following search terms: {all_search_terms}.

    The following criteria were used to evaluate the search results obtained:
    1. The following common search results should be DISAPPROVED: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations of the 'best' products, product rankings.
    2. The following common search results should be APPROVED: product listings with prices, product catalogs with prices, list of multiple product models with prices.

    Based on the evaluation of the search results, the following search terms were approved: {approved_search_terms}.
    The following search terms were disapproved: {disapproved_search_terms}.'''
    
    USER_REFINEMENT_PROMPT = f'''Based on the information given to you in the system message, provide {n_refined_search_terms} new refined search terms that you think would yield more relevant results.
    Do NOT repeat any of the search terms used so far, regardless of whether they were approved.'''

    # Messages for the model
    messages = [
        SystemMessage(content=SYSTEM_REFINEMENT_PROMPT),
        HumanMessage(content=USER_REFINEMENT_PROMPT)
    ]

    # Model with structured output
    structured_model = search_term_refinement_model.with_structured_output(RefinedSearchTerms)

    # Invoke model
    response = structured_model.invoke(messages)

    # Make sure that the refined search terms have not been used before
    refined_search_terms = response.refined_search_terms
    refined_search_terms = [term for term in refined_search_terms if term not in all_search_terms]

    # Print the refined search terms that were discarded
    if len(response.refined_search_terms) != len(refined_search_terms):
        print("The following refined search terms were discarded because they were already used: ", [term for term in response.refined_search_terms if term in all_search_terms])

    print("Refined search terms: ", refined_search_terms)
    
    return {"refined_search_terms": refined_search_terms}


# SEARCH NODE
def search_node(state: AgentState):

    # Retrieve from the state
    refined_search_terms = state.get('refined_search_terms', [])
    max_results_per_search_term = state.get('max_results_per_search_term', 3) # default is 3
    
    # Create empty list of dictionaries to store search results
    search_results = []

    # Initiate the Tavily client
    from tavily import TavilyClient
    tavily_client = TavilyClient() # API key is stored in the environment variable

    # Make an API call to Tavily
    for term in refined_search_terms:
        print('Searching for:', term)
        response = tavily_client.search(
                query=term, 
                max_results=max_results_per_search_term,
                include_raw_content=True
                )
        print("Amount of results found for this search term: ", len(response['results']))
        for result in response['results']:
            result['query'] = term
            search_results.append(result)

    print("Amount of search results added for evaluation: ", len(search_results))
    # print('AUX. These are the search results to evaluate:', search_results)

    return {"search_results_to_evaluate": search_results}


#%% CONDITIONAL GRAPH EDGES

# Check if minimum number of approved search results is reached
def count_approved_search_results(state):

    # Get relevant elements from state
    evaluated_search_results = state.get('evaluated_search_results', [])
    min_approved_search_results = state.get('min_approved_search_results', 1)
    iteration_number = state.get('iteration_number')
    print("Just finished iteration number: ", iteration_number)
    max_iterations = state.get('max_iterations', 2)

    approved_search_results = [evaluated_result for evaluated_result in evaluated_search_results if evaluated_result['evaluation_outcome']==True]
    print(f"Count of approved search results: {len(approved_search_results)}")

    # Decide whether to continue or end the pipeline
    if len(approved_search_results) >= min_approved_search_results:
        print("Minimum number of approved search results reached. End of pipeline.")
        print("Final approved search results: ")
        for approved_result in approved_search_results:
            print(approved_result['title'], approved_result['url'])
        return END
    
    elif iteration_number >= max_iterations:
        print("Maximum number of iterations reached. End of pipeline.")
        print("Final approved search results: ")
        for approved_result in approved_search_results:
            print(approved_result['title'], approved_result['url'])
        return END
    
    else:
        print("Minimum number of approved search results not reached. Continue to next iteration.")
        return 'refine_search_terms'


#%% DEFINE THE GRAPH

from langgraph.graph import StateGraph, END

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
    {END: END, 'refine_search_terms': 'refine_search_terms'} # !!! Fix this to expand the graph
)

# # Finish point
# builder.set_finish_point('evaluator_of_search_results')

#%% COMPILE THE GRAPH

memory = SqliteSaver.from_conn_string(':memory:')
graph = builder.compile(checkpointer=MemorySaver())


#%% OPTION TO RETURN GRAPH
def get_graph():
    return graph


#%% VISUALIZE THE GRAPH
def visualize_graph():
    graph = get_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))


#%% CALCULATE COSTS
def calculate_costs(
        search_results_to_evaluate, commodity, 
        min_similarity_score=None, use_raw_content=True, include_justification=True,
        justification_length=300, n_refined_search_terms=3, max_iterations=3,
        refined_search_term_length=60, max_results_per_search_term=3, verbose=True,
        price_1m_input_tokens=0.075, price_1m_output_tokens=0.3, tavily_search_price=0.02
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
        else:
            search_result_content_to_evaluate['content'] = search_result['content']

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


    # SEARCH-TERM-REFINMENT NODE

    # Retrieve the search terms that received approval and disapproval
    all_search_terms = [evaluated_result['query'] for evaluated_result in search_results_to_evaluate]
    all_search_terms = list(set(all_search_terms))

    # Divide all_search_terms into two lists with equal amount of elements
    approved_search_terms = all_search_terms[:len(all_search_terms)//2]
    disapproved_search_terms = all_search_terms[len(all_search_terms)//2:]

    # Define the prompt
    SYSTEM_REFINEMENT_PROMPT = f'''You are an expert analyst tasked with refining search terms to improve the quality of search results for online sources of product prices for the following category of products: {commodity}.

    An initial search was conducted using the following search terms: {all_search_terms}.

    The following criteria were used to evaluate the search results obtained:
    1. The following common search results should be DISAPPROVED: appraisal tools, product valuation calculators, general information about the product, user manuals, recommendations of the 'best' products, product rankings.
    2. The following common search results should be APPROVED: product listings with prices, product catalogs with prices, list of multiple product models with prices.

    Based on the evaluation of the search results, the following search terms were approved: {approved_search_terms}.
    The following search terms were disapproved: {disapproved_search_terms}.'''
    
    USER_REFINEMENT_PROMPT = f'''Based on the information given to you in the system message, provide {n_refined_search_terms} refined search terms that you think would yield more relevant results.
    Do NOT repeat any of the search terms used so far, regardless of whether they were approved.'''

    for iteration in range(max_iterations):

        # Input tokens
        input_tokens_refinement_node += len(SYSTEM_REFINEMENT_PROMPT)
        # In each new iteration, n_refined_search_terms additional search terms could be added
        # In the first iteration, there is no need to add anything else
        # In the system prompt, they would count as two
        input_tokens_refinement_node += refined_search_term_length*iteration

        # User input
        input_tokens_refinement_node += len(USER_REFINEMENT_PROMPT)

        # Output tokens
        output_tokens_refinement_node += n_refined_search_terms*refined_search_term_length


    # TAVILY SEARCH NODE
    tavily_search_count = 0
    tavily_search_count += max_iterations-1

    # Divide characters by 4 to get tokens
    input_tokens_evaluation_node = round(input_tokens_evaluation_node / 4, 0)
    output_tokens_evaluation_node = round(output_tokens_evaluation_node / 4, 0)
    input_tokens_refinement_node = round(input_tokens_refinement_node / 4, 0)
    output_tokens_refinement_node = round(output_tokens_refinement_node / 4, 0)

    total_input_tokens = input_tokens_evaluation_node + input_tokens_refinement_node
    total_output_tokens = output_tokens_evaluation_node + output_tokens_refinement_node

    total_cost_input_tokens = total_input_tokens*(price_1m_input_tokens/1e6)
    total_cost_output_tokens = total_output_tokens*(price_1m_output_tokens/1e6)

    total_cost_tokens = total_cost_input_tokens + total_cost_output_tokens

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

        print("Total number of Tavily searches: ", tavily_search_count)

    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_token_cost': total_cost_tokens,
        'tavily_search_count': tavily_search_count
    }

# #%% 

# if __name__ == "__main__":
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
