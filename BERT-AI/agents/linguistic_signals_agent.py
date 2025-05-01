from agents.base_agent import BaseStanceAgent

class LinguisticSignalsAgent(BaseStanceAgent):
    def __init__(self):
        super().__init__(
            role="Linguistic Nuance Detector",
            goal="Identify and analyze subtle linguistic markers that reveal underlying stance and emotional positioning.",
            backstory="""
            As an advanced linguistic intelligence specializing in pragmatic analysis and discourse interpretation, 
            you possess a unique ability to decode subtle communicative mechanisms. Your expertise lies in detecting 
            micro-level linguistic cues that reveal an author's true stance, including modal verbs, hedging language, 
            and implicit sentiment markers. Your analytical framework includes:
            
            - Advanced pragmatic interpretation
            - Detection of microscopic linguistic markers
            - Analysis of modal verbs and hedging language
            - Extraction of implicit meanings
            - Mapping of sophisticated discourse strategies
            
            You approach language as a complex, multi-layered system, where every linguistic choice reveals intricate 
            cognitive and attitudinal positioning.
            """,
            model="gemini-2.0-flash-lite"
        )
    
    def detect_linguistic_signals(self, text, topic):
        """
        Analyze linguistic signals that indicate stance in a given text.
        """
        prompt = f"""
        Conduct a comprehensive linguistic analysis of the following text in the context of the topic '{topic}':

        *Text:* {text}

        Focus on the following aspects:
        
        1. *Modal Verb Usage:*
           - Identify all modal verbs (e.g., might, should, must, could) and analyze their epistemological and attitudinal implications.
           
        2. *Hedging Language & Epistemic Markers:*
           - Detect subtle qualifiers and uncertainty markers, explaining their impact on stance positioning.
           
        3. *Emotional Intensity & Affective Language:*
           - Identify emotionally charged words and phrases, assessing their contribution to the expressed stance.
           
        4. *Implied Meaning & Subtext:*
           - Uncover hidden connotations and indirect communicative strategies that influence interpretation.
           
        5. *Linguistic Stance Indicators:*
           - Analyze syntactic structures, lexical choices, and discourse patterns that reveal underlying stance.

        Provide a detailed, multi-layered analysis, explaining how each identified linguistic feature contributes to the stance expression.
        """
        
        # Send message and get response
        response = self.chat.send_message(prompt)
        return response.text