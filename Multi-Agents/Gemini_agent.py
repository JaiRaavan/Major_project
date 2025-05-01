import pandas as pd
import time
import logging
import google.generativeai as genai
import time

# Your Gemini API key
genai.configure(api_key="AIzaSyBYzVxDZfdW1znCbUtVcalm9SK0HkMhWqw")  # Replace with your Gemini API key

# Create model once
model = genai.GenerativeModel("gemini-1.5-flash")

# Target-role mapping
target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist"
}

def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")


def get_completion(prompt):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            time.sleep(5)  # Wait 5 seconds after each successful call
            return response.text.strip()
        except Exception as e:
            logging.warning(f"Retry {attempt + 1}/{max_retries} failed: {str(e)}")
            time.sleep(10)  # Wait before retrying


def get_completion_with_role(role, instruction, content):
    prompt = f"You are a {role}.\n{instruction}\n{content}"
    return get_completion(prompt)

def linguist_analysis(tweet):
    instruction = "Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else."
    return get_completion_with_role("linguist", instruction, tweet)

def expert_analysis(tweet, target):
    role = target_role_map.get(target, "expert")
    instruction = f"Accurately and concisely explain the key elements contained in the quote, such as characters, events, parties, religions, etc. Also explain their relationship with {target} (if exist). Do nothing else."
    return get_completion_with_role(role, instruction, tweet)

def user_analysis(tweet):
    instruction = "Analyze the following sentence, focusing on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else."
    return get_completion_with_role("heavy social media user", instruction, tweet)

def stance_analysis(tweet, ling_response, expert_response, user_response, target, stance):
    role = target_role_map.get(target, "expert")
    prompt = f"'''{tweet}'''\n <<<{ling_response}>>>\n [[[{expert_response}]]]\n---{user_response}---\n" \
             f"You think the attitude behind the sentence surrounded by ''' ''' is {stance} of {target}. " \
             f"The content enclosed by <<< >>> represents linguistic analysis. The content within [[[ ]]] represents the analysis of a {role}. " \
             f"The content enclosed by --- --- represents the analysis of a heavy social media user. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion."
    return get_completion(prompt)

def final_judgement(tweet, favor_response, against_response, target):
    prompt = f"Determine whether the sentence is in favor of or against {target}, or is irrelevant to {target}.\n" \
             f"Sentence: {tweet}\nJudge this in relation to the following arguments:\n" \
             f"Arguments that the attitude is in favor: {favor_response}\n" \
             f"Arguments that the attitude is against: {against_response}\n" \
             f"Choose from:\n A: Against\nB: Favor\nC: Irrelevant\n" \
             f"Constraint: Answer with only the option above that is most accurate and nothing else."
    judgement = get_completion(prompt)
    print(judgement)
    return judgement


def add_predictions_sequential(data):
    results = []
    counter = 0
    for index, row in data.iterrows():
        tweet = row['Tweet']
        target = row['Target']
        # original_stance = row['Stance']

        counter += 1
        if counter < 121:
            continue
       
        if counter % 2 == 0 and counter != 0:
            time.sleep(60)

        # Get model responses
        ling_response = linguist_analysis(tweet)
        expert_response = expert_analysis(tweet, target)
        user_response = user_analysis(tweet)

        favor_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "in favor")
        against_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "against")
        # neutral_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "neutral")
        final_response = final_judgement(tweet, favor_response, against_response, target)

        # Write result directly into DataFrame
        data.at[index, 'Final Judgement'] = final_response

        # Save every 20 iterations
        if counter % 20 == 0 and counter != 0:
            print(f"Saving after {counter} iterations...")
            data.to_csv(f"./results/test_biden_Gemini_{counter}.csv", index=False)


    # Save final version
    data.to_csv("./results/test_biden_Gemini.csv", index=False)


if __name__ == "_main_":
    data = load_csv_data("test_biden.csv")
    add_predictions_sequential(data)