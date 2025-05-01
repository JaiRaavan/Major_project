import os
import traceback
import re
import google.generativeai as genai
from dotenv import load_dotenv
from agents.base_agent import BaseStanceAgent
from agents.sentiment_agent import SentimentAgent
from agents.argumentative_structure_agent import ArgumentativeStructureAgent
from agents.linguistic_signals_agent import LinguisticSignalsAgent
from agents.validation_agent import ValidationAgent

load_dotenv()

class StanceCoordinatorAgent(BaseStanceAgent):
    def __init__(self, 
                 sentiment_agent=None, 
                 argumentative_agent=None, 
                 linguistic_agent=None,
                 validation_agent=None):
        # Configure Google Generative AI
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Initialize agents if not provided
        self._sentiment_agent = sentiment_agent or SentimentAgent()
        self._argumentative_agent = argumentative_agent or ArgumentativeStructureAgent()
        self._linguistic_agent = linguistic_agent or LinguisticSignalsAgent()
        self._validation_agent = validation_agent or ValidationAgent()
        
        # Use Gemini Pro for coordination
        self.llm = genai.GenerativeModel('gemini-2.0-flash-lite')
        self.chat = self.llm.start_chat(history=[])
        
        # Maximum retries for validation
        self.max_retries = 2
        
        super().__init__(
            role="Enhanced Stance Synthesis Strategist (ESSS-1)",
            goal="Synthesize insights from multiple analytical agents and validate stance determination for accuracy",
            backstory="""
            You are an expert in stance analysis, integrating insights from sentiment analysis, argumentative structure, 
            and linguistic signals to determine the stance expressed in a text. Your expertise lies in:
            
            - Analyzing multi-dimensional textual data
            - Synthesizing insights from specialized agents
            - Identifying subtle linguistic and rhetorical cues
            - Providing well-reasoned stance determinations
            - Validating stance predictions for accuracy and consistency
            
            Treat stance detection as an intricate analytical process where multiple perspectives converge to reveal 
            the underlying communicative intent, with additional validation to ensure reliability.
            """,
            model="gemini-2.0-flash-lite"
        )
    
    def _extract_stance(self, synthesis_text):
        """
        Extract the final stance from the synthesis text.
        
        Args:
            synthesis_text (str): Full synthesis text.
        
        Returns:
            str: Extracted stance (Favour/Against/Neutral).
        """
        stances = ['Favour', 'Against', 'Neutral']
        for stance in stances:
            if re.search(rf'Final\s*Stance\s*:\s*{stance}', synthesis_text, re.IGNORECASE):
                return stance
        for stance in stances:
            if stance.lower() in synthesis_text.lower():
                return stance
        return 'Neutral'
    
    def _normalize_stance(self, stance):
        """
        Normalize stance labels between different formats (Favour/Favor, etc.)
        
        Args:
            stance (str): The stance to normalize
            
        Returns:
            str: Normalized stance
        """
        stance = stance.lower()
        if stance in ['favour', 'favor']:
            return 'Favor'
        elif stance == 'against':
            return 'Against'
        else:
            return 'Neutral'
    
    def synthesize_stance(self, text, topic, retry_count=0):
        """
        Coordinate multiple agents to determine stance with validation.
        
        Args:
            text (str): The text to analyze
            topic (str): The topic to determine stance on
            retry_count (int): Current retry attempt (for internal use)
            
        Returns:
            dict: Final stance analysis result with validation information
        """
        try:
            print(f"\n🔍 STANCE DETECTION PROCESS (Attempt {retry_count + 1}) 🔍")
            print("=" * 50)
            
            # Clear the chat history for retries to avoid contamination
            if retry_count > 0:
                # Reset the chat session to start fresh
                self.chat = self.llm.start_chat(history=[])
            
            # Sentiment Analysis
            print("\n📊 SENTIMENT ANALYSIS:")
            try:
                sentiment_analysis = self._sentiment_agent.sentiment_analysis(text, topic)
                print("Sentiment Insights:", sentiment_analysis)
            except Exception as e:
                print(f"Sentiment Analysis Error: {e}")
                sentiment_analysis = "Error performing sentiment analysis."
            
            # Argumentative Structure Analysis
            print("\n🧠 ARGUMENTATIVE STRUCTURE ANALYSIS:")
            try:
                argumentative_analysis = self._argumentative_agent.analyze_argumentative_structure(text, topic)
                print("Argumentative Structure Insights:", argumentative_analysis)
            except Exception as e:
                print(f"Argumentative Analysis Error: {e}")
                argumentative_analysis = "Error analyzing argumentative structure."
            
            # Linguistic Signals Analysis
            print("\n🔤 LINGUISTIC SIGNALS ANALYSIS:")
            try:
                linguistic_analysis = self._linguistic_agent.detect_linguistic_signals(text, topic)
                print("Linguistic Signals Insights:", linguistic_analysis)
            except Exception as e:
                print(f"Linguistic Analysis Error: {e}")
                linguistic_analysis = "Error detecting linguistic signals."
            
            # Stance Synthesis
            print("\n🤔 STANCE SYNTHESIS:")
            
            # Add feedback from previous validation attempts if this is a retry
            additional_guidance = ""
            if retry_count > 0:
                additional_guidance = f"""
                Previous stance analysis attempts were incorrect according to validation.
                Please carefully consider all evidence before determining the stance.
                Pay particular attention to the intensity and directness of sentiment expressions.
                """
            
            synthesis_prompt = f"""
            You are an expert stance analyst tasked with determining the stance expressed in the given text.
            
            *Context:*
            - *Text:* {text}
            - *Topic:* {topic}
            
            *Analytical Inputs:*
            1. *Sentiment Analysis:* {sentiment_analysis}
            2. *Argumentative Structure:* {argumentative_analysis}
            3. *Linguistic Signals:* {linguistic_analysis}
            
            {additional_guidance}
            
            *Your Task:*
            Carefully synthesize these insights to determine the most accurate stance (Favour, Against, or Neutral).
            Provide a clear and concise response including:
            - The *Final Stance* in the format: *Final Stance: [Favour/Against/Neutral]*
            - A brief *justification* for why this stance was chosen.
            - Key *indicators* from sentiment, argumentative structure, and linguistic analysis that support this conclusion.
            """
            
            try:
                response = self.chat.send_message(synthesis_prompt)
                print("\n✨ FINAL STANCE SYNTHESIS:")
                print(response.text)
                coordinator_stance = self._extract_stance(response.text)
                print(f"\n🎯 Initial Stance Determination: {coordinator_stance}")
                
                # Normalize the stance for validation
                normalized_stance = self._normalize_stance(coordinator_stance)
                
                # Validate the stance
                print("\n⚖️ VALIDATING STANCE...")
                validation_result = self._validation_agent.validate_stance(text, topic, normalized_stance)
                
                # Process validation result
                if isinstance(validation_result, dict):
                    validation_stance_correct = validation_result.get('stance_correct', False)
                    validation_stance = validation_result.get('correct_stance', 'Neutral')
                    validation_reasoning = validation_result.get('reasoning', 'No reasoning provided')
                    
                    # Handle error case explicitly
                    if 'error' in validation_result:
                        print(f"Validation Error: {validation_result['error']}")
                        validation_reasoning = f"Error: {validation_result['error']}"
                else:
                    # Simple string parsing if not properly formatted JSON
                    validation_text = str(validation_result)
                    validation_stance_correct = 'true' in validation_text.lower()
                    validation_stance = 'Neutral'
                    if 'correct_stance' in validation_text:
                        for stance in ['Favor', 'Against', 'Neutral']:
                            if stance in validation_text:
                                validation_stance = stance
                                break
                    validation_reasoning = validation_text
                
                print(f"Validation Result: Stance Correct = {validation_stance_correct}")
                print(f"Validation Suggested Stance: {validation_stance}")
                print(f"Validation Reasoning: {validation_reasoning}")
                
                # MODIFIED LOGIC: If validation and synthesis stances don't match, 
                # always take the validation stance as final stance
                if not validation_stance_correct:
                    print("\n⚠️ SYNTHESIS AND VALIDATION STANCES DON'T MATCH")
                    print(f"Taking validation stance ({validation_stance}) as final stance")
                    final_stance = validation_stance
                    
                    # Skip retries since we're already accepting the validation stance
                    retry_needed = False
                else:
                    # Validation confirms synthesis stance
                    final_stance = normalized_stance
                    retry_needed = False
                
                # Format the final result
                final_result = {
                    "text": text,
                    "topic": topic,
                    "coordinator_stance": normalized_stance,
                    "validation_stance": validation_stance,
                    "validation_correct": validation_stance_correct,
                    "final_stance": final_stance,
                    "analysis_rounds": retry_count + 1,
                    "coordinator_analysis": response.text,
                    "validation_reasoning": validation_reasoning
                }
                
                # Final output
                print("\n🏆 FINAL VALIDATED STANCE:")
                print(f"Final Stance: {final_stance}")
                print(f"Analysis required {retry_count + 1} round(s)")
                
                return final_result
                
            except Exception as synthesis_error:
                print(f"Stance Synthesis Error: {synthesis_error}")
                return {"final_stance": "Neutral", "error": str(synthesis_error)}
        
        except Exception as e:
            print(f"Comprehensive Error in Stance Detection: {e}")
            traceback.print_exc()
            return {"final_stance": "Neutral", "error": str(e)}
    
    # Getters for agents
    @property
    def sentiment_agent(self):
        return self._sentiment_agent
    
    @property
    def argumentative_agent(self):
        return self._argumentative_agent
    
    @property
    def linguistic_agent(self):
        return self._linguistic_agent
    
    @property
    def validation_agent(self):
        return self._validation_agent