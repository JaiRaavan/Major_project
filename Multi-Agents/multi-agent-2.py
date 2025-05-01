# importing the necessary modules
import json
import pandas as pd
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.tools import StructuredTool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# Response type---------------------------------------------
class StanceDetectionOutput(BaseModel):
    stance: str 
    sentiment_score: int
    context_analysis: str  
    keywords: List[str] 
    

# Create Linguistic Agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Create a parser for structured output
parser = PydanticOutputParser(pydantic_object=StanceDetectionOutput)


# Prompts --------------------------------------------

# 1. Linguistic Expert (Analyzes Grammar, Tone, Rhetoric)
linguistic_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a linguist. Accurately and concisely explain the linguistic elements in the sentence 
        and how these elements affect meaning, including grammatical structure, tense and inflection, 
        rhetorical devices, and lexical choices.
        Do nothing else.
    """),
    ("human", "{tweet}"),
    # ("placeholder", "{agent_scratchpad}")
])

# linguistic_agent = create_tool_calling_agent(llm=llm, tools=[], prompt=linguistic_prompt)
# linguistic_executor = AgentExecutor(agent=linguistic_agent,tools=[], verbose=True)


# 2. Domain Specialist (Understands Political, Social, and Historical Context)
domain_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a domain expert. Accurately and concisely explain the key elements contained in the quote, 
        such as characters, events, parties, religions, etc. Also, explain their relationship with {target} (if they exist).
        Do nothing else.
    """),
    ("human", "{tweet}"),
    # ("placeholder", "{agent_scratchpad}")
])

# domain_agent = create_tool_calling_agent(llm=llm, tools=[], prompt=domain_prompt)
# domain_executor = AgentExecutor(agent=domain_agent,tools=[], verbose=True)



# 3. Social Media Veteran (Understands Hashtags, Emojis, and Online Trends)
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

social_media_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a heavy social media user and are very familiar with the way of expression on the Internet. 
        Analyze the following sentence, focusing on the content, emotional tone, implied meaning, and so on. 
        Do nothing else.
    """),
    ("human", "{tweet}"),
    ("placeholder", "{agent_scratchpad}")
])
social_media_agent = create_tool_calling_agent(llm=llm, tools=[search_tool], prompt=social_media_prompt)
social_media_executor = AgentExecutor(agent=social_media_agent, tools=[search_tool], verbose=True)



# 4. debating agent argues for a specific stance (Favor, Against, Neutral).
debate_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Tweet: {tweet}. 
        Linguistic analysis: {linguistic_response}.
        Domain analysis: {domain_response}. 
        Social media analysis: {social_media_response}.
        
        You think the attitude behind the tweet is {stance} of {target}. 
        Identify the top three pieces of evidence from the analyses that best support your opinion and argue for your opinion.
    """),
    ("human", "{tweet}"),
    # ("placeholder", "{agent_scratchpad}")
])
# debate_agent = create_tool_calling_agent(llm=llm, tools=[], prompt=debate_prompt)
# debate_executor = AgentExecutor(agent=debate_agent, tools=[], verbose=True)


# 5. judge agent evaluates all debates and makes the final decision.
final_judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Determine whether the sentence is in favor of, against, or neutral towards {target}.
        
        Sentence: {tweet}

        Consider the following arguments:
        - Arguments that the attitude is in favor: {favor_response}
        - Arguments that the attitude is against: {against_response}
        - Arguments that the attitude is neutral: {neutral_response}

        Choose from:
        A: Against
        B: Favor
        C: Neutral
        
        if Answer is Against sentiment score must be -1 and stance will be AGAINST
        if Answer is Favor sentiment score must be 1 and stance will be FAVOR
        if Answer is Neutral sentiment score must be 0 and stance will be NEUTRAL

        **Constraint**: Answer with only the option above that is most accurate and nothing else.
        
        Your response **must follow this structured format**:
            {format_instructions}
    """),
    ("human", "{tweet}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

judge_agent = create_tool_calling_agent(llm=llm, tools=[], prompt=final_judge_prompt)
judge_executor = AgentExecutor(agent=judge_agent, tools=[], verbose=True)




# Actual Game Starts Here ------------------------------------

# Load the dataset
csv_file = "./data/val.csv"  # Replace with the actual CSV file path
df = pd.read_csv(csv_file)

# Ensure required columns exist
if "Tweet" not in df.columns or "Target" not in df.columns:
    raise ValueError("CSV file must contain 'tweet' and 'target' columns.")

# Storage for results
results = []

# Process each row in the dataset
for index, row in df.iterrows():
    print(f"-----------------------------{index}-----------------------------")
    tweet = row["Tweet"]
    target = row["Target"]

    # Step 1: Run expert agents
    formatted_prompt = linguistic_prompt.format(tweet=tweet)
    linguistic_response = llm.invoke(formatted_prompt)
    
    formatted_prompt = domain_prompt.format(tweet=tweet, target=target)
    domain_response = llm.invoke(formatted_prompt)
    
    social_media_response = social_media_executor.invoke({"tweet": tweet})

    # Step 2: Run debating agents for each stance
    formatted_prompt = debate_prompt.format(tweet= tweet,
        target= target,
        linguistic_response= linguistic_response,
        domain_response= domain_response,
        social_media_response= social_media_response,
        stance= "Favor"
    )
    favor_response = llm.invoke(formatted_prompt)
    
    formatted_prompt = debate_prompt.format(tweet= tweet,
        target= target,
        linguistic_response= linguistic_response,
        domain_response= domain_response,
        social_media_response= social_media_response,
        stance= "Against"
    )

    against_response = llm.invoke(formatted_prompt)
    
    formatted_prompt = debate_prompt.format(tweet= tweet,
        target= target,
        linguistic_response= linguistic_response,
        domain_response= domain_response,
        social_media_response= social_media_response,
        stance= "Neutral"
    )

    neutral_response = llm.invoke(formatted_prompt)

    # Step 3: Run final judge agent
    final_stance = judge_executor.invoke({
        "tweet": tweet,
        "target": target,
        "favor_response": favor_response,
        "against_response": against_response,
        "neutral_response": neutral_response
    })

    output = final_stance['output'][0]['text']

    # Extract additional information if available
    # Convert string to dictionary
    data = json.loads(output)
    # Extract required values
    predicted_stance = data["stance"]
    sentiment_score = data["sentiment_score"]
    keywords = data["keywords"]
    actual_stance = row['Stance']  # If available in CSV

    # Store results
    results.append({
        "tweet": tweet,
        "target": target,
        "actual_stance": actual_stance,
        "predicted_stance": predicted_stance,
        "sentiment_score": sentiment_score,
        "keywords": keywords
    })

# Save results to a new CSV file
output_df = pd.DataFrame(results)
output_df.to_csv("./outputs/processed_tweets.csv", index=False)

print("Processing complete. Results saved to 'processed_tweets.csv'.")