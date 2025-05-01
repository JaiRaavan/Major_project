from agents.base_agent import BaseStanceAgent

class ArgumentativeStructureAgent(BaseStanceAgent):
    def __init__(self):
        super().__init__(
            role="Argumentative Discourse Analyst",
            goal="Deconstruct the logical structure and rhetorical strategies used in the text to identify underlying stance.",
            backstory="""
            You are a world-renowned expert in rhetoric and argumentation, holding a PhD in Discourse Analysis. 
            Your specialization lies in deconstructing complex arguments, identifying logical fallacies, and mapping 
            persuasive techniques used in communication. Your expertise includes:
            
            - Advanced rhetorical strategy deconstruction
            - Logical argument mapping
            - Persuasive technique identification
            - Bias and logical fallacy detection
            - Nuanced argumentative stance inference
            
            You analyze each text as a carefully constructed rhetorical landscape, where arguments are not merely 
            sequences of statements but strategic mechanisms revealing deeper perspectives.
            """,
            model="gemini-2.0-flash-lite"
        )
    
    def analyze_argumentative_structure(self, text, topic):
        """
        Analyze the argumentative structure and rhetorical strategies of the text.
        """
        prompt = f"""
        Conduct an in-depth analysis of the argumentative structure in the following text, focusing on the topic: '{topic}'.

        *Text:* {text}

        Your analysis should include:
        
        1. *Identification of the Main Argumentative Thesis*
           - Clearly define the central claim being presented.
           
        2. *Evaluation of Supporting Arguments*
           - Break down the supporting points and their effectiveness.
           
        3. *Rhetorical Strategies & Persuasive Techniques*
           - Identify and assess the persuasive methods used (e.g., ethos, pathos, logos).
           
        4. *Logical Structure & Coherence*
           - Examine the logical flow and organization of arguments.
           
        5. *Detection of Bias & Persuasion Methods*
           - Highlight any implicit biases or persuasive framing techniques.
           
        6. *Analysis of Supporting & Counterarguments*
           - Identify counterarguments presented (or the lack thereof) and their treatment.
           
        7. *Inference of Likely Stance*
           - Deduce the overall stance conveyed through argumentative construction.
           
        Provide a deep and structured breakdown, including specific textual evidence to support each analysis point.
        """
        
        # Send message and get response
        response = self.chat.send_message(prompt)
        return response.text