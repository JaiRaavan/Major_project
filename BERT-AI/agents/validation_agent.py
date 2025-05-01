from agents.base_agent import BaseStanceAgent
from openai import OpenAI
import json
import re

class ValidationAgent(BaseStanceAgent):
    def __init__(self):
        super().__init__(
            role="Stance Validation Expert",
            goal="Validate whether a predicted stance logically aligns with the text in relation to a target topic using linguistic and emotional cues.",
            backstory="""
            As a stance validation expert with a background in linguistic logic and sentiment-context mapping, your job is to assess 
            whether a predicted stance accurately reflects the attitude conveyed in a piece of text towards a specific target. 
            You rely on rule-based reasoning, contextual interpretation, and emotional cue tracking to detect mismatches.
            
            Your core validation principles include:
            - Sentiment polarity should align with stance: Positive → Favor, Negative → Against, Neutral → Neutral.
            - Sentiment intensity matters: If the sentiment is weak, the stance is likely Neutral.
            - Target relevance is crucial: If the text doesn't refer to the target, stance is likely Neutral or Invalid.
            - Linguistic cues (supportive vs. critical words) should match the stance.
            - Contradictory expressions or sarcasm can imply ambiguity or mismatch.
            """,
            model="deepseek/deepseek-chat-v3-0324:free"
        )
        
        # Store model name for reference
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        
        # Initialize the OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        
        # Define model parameters to use with API calls
        self.model_params = {
            "temperature": 0.7,
            "max_tokens": 1024
        }
    
    def _extract_json_from_response(self, text):
        """
        Try to extract JSON from the response text.
        Sometimes the model includes additional text around the JSON.
        """
        # Try to find JSON pattern in the text
        json_pattern = r'({.*})'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            potential_json = match.group(1)
            try:
                return json.loads(potential_json)
            except:
                pass
        
        # If no valid JSON found, attempt to extract stance information from text
        stance_result = {
            "stance_correct": False,
            "correct_stance": "Neutral",
            "reasoning": "Failed to parse response"
        }
        
        # Look for stance_correct indicators
        if re.search(r'(stance_correct|stance correct).*true', text, re.IGNORECASE):
            stance_result["stance_correct"] = True
        
        # Look for correct_stance indicators
        for stance in ["Favor", "Against", "Neutral"]:
            if re.search(rf'(correct_stance|correct stance).*{stance}', text, re.IGNORECASE):
                stance_result["correct_stance"] = stance
                break
        
        # Extract some reasoning if possible
        reasoning_match = re.search(r'(reasoning|reasoning:|REASONING).*', text, re.IGNORECASE)
        if reasoning_match:
            stance_result["reasoning"] = reasoning_match.group(0)
        else:
            stance_result["reasoning"] = text[:500]  # Use part of the text as reasoning
            
        return stance_result
        
    def validate_stance(self, text, target, predicted_stance):
        """
        Validate the stance prediction using DeepSeek's reasoning capabilities
        through the OpenRouter API.
        """
        prompt = f"""
        You are a Stance Validation Expert with strong reasoning abilities. Analyze whether the predicted stance accurately reflects the author's 
        position towards the given target in the text. Apply strict linguistic and sentiment-based reasoning rules.
        
        *TEXT:* 
        {text}
        *TARGET:* {target}
        
        *PREDICTED STANCE:* {predicted_stance}
        
        *VALIDATION RULES TO FOLLOW:*
        1. If the text shows *positive sentiment* towards the target, the correct stance is *Favor*.
        2. If the text shows *negative sentiment* towards the target, the correct stance is *Against*.
        3. If the text shows *no strong sentiment* or is unrelated to the target, the stance should be *Neutral*.
        4. Consider the *intensity of emotion*; weak sentiment (e.g., minor opinions) should not be classified as Favor/Against.
        5. Ensure the target is *actually mentioned or referred to* in the text.
        6. Watch for *contradictions, sarcasm, or mixed expressions*, which may suggest a neutral or ambiguous stance.
        7. Validate that the *language used (keywords, tone, expressions)* supports the stance.
        
        *REASONING APPROACH:*
        - First, carefully identify all mentions or references to the target in the text
        - Next, analyze the sentiment associated with each reference (positive, negative, neutral)
        - Then, assess the overall intensity and consistency of the sentiment
        - Finally, determine if the predicted stance matches your analysis
        
        *OUTPUT FORMAT (Strict JSON):*
        {{
            "stance_correct": true or false,
            "correct_stance": "Favor" or "Against" or "Neutral",
            "reasoning": "Step-by-step justification using the rules above."
        }}
        
        IMPORTANT: Your FULL RESPONSE should be ONLY valid JSON, exactly as specified in the format above. Do not include any additional text, markdown, or explanations - just the JSON object.
        give the response only in english language
        """
        
        # Create messages structure for the API call
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Use the updated OpenRouter API pattern with model parameters
            completion = self.client.chat.completions.create(
                extra_body={},
                model=self.model,
                messages=messages,
                temperature=self.model_params["temperature"],
                max_tokens=self.model_params["max_tokens"]
            )
            
            # Extract the content from the response
            result = completion.choices[0].message.content
            
            print("Validator Response:", result)
            
            # Parse the JSON if it's returned as a string
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                print("Failed to parse JSON response, attempting to extract JSON manually")
                return self._extract_json_from_response(result)
                
        except Exception as e:
            error_message = str(e)
            print(f"Validation error occurred: {error_message}")
            
            # Return a fallback result
            return {
                "stance_correct": False,  # Default to not correct, let the system use its own judgment
                "correct_stance": "Neutral",  # Default to neutral as the safest option
                "reasoning": f"Error in validation: {error_message}"
            }