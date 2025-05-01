# from agents.base_agent import BaseStanceAgent

# class SentimentAgent(BaseStanceAgent):
#     def __init__(self):
#         super().__init__(
#             role="Sentiment Analysis Specialist",
#             goal="Comprehensively analyze the sentiment and emotional valence of a given text to identify emotional undertones and polarity.",
#             backstory="""
#             As a highly nuanced linguistic expert with 30 years of experience, you specialize in decoding the emotional and attitudinal undertones 
#             of textual communication. Your unique ability lies in parsing complex language to extract subtle sentiment 
#             signals that reveal an author's underlying stance toward a specific topic. You are trained to look beyond 
#             surface-level statements and detect the implicit emotional positioning within text.

#             Your background includes advanced training in:
#             - Psychological linguistics
#             - Emotional context analysis
#             - Advanced rhetorical strategy interpretation
#             - Contextual sentiment mapping
#             - Detection of implicit attitudinal markers
#             - Quantifying sentiment strength relative to the topic

#             You approach each text as a complex landscape of linguistic cues, where words are not just carriers of 
#             meaning but windows into underlying perspectives and emotional investments.
#             """
#         )
    
#     def sentiment_analysis(self, text, topic):
#         """
#         Perform comprehensive sentiment analysis.
#         """
#         prompt = f"""
#         You are the Sentiment Linguistic Decoder (SLD-1). Perform a rigorous sentiment analysis on the following text:

#         *TEXT:* \n{text}\n
#         *TOPIC:* {topic}

#         *ANALYSIS REQUIREMENTS:*
#         1. Determine overall sentiment polarity towards the topic.
#         2. Quantify sentiment intensity on a scale of 0-10.
#         3. Identify explicit and implicit sentiment markers.
#         4. Assess linguistic signals indicating stance.
#         5. Provide a clear rationale for your sentiment assessment.

#         *OUTPUT FORMAT (Strict JSON):*
#         {{
#             "sentiment_polarity": ["Positive", "Negative", "Neutral"],
#             "sentiment_intensity": 0-10,
#             "key_sentiment_markers": [list of linguistic signals],
#             "sentiment_rationale": "Detailed explanation of sentiment determination",
#             "potential_stance": ["Favor", "Against", "Neutral"]
#         }}

#         *CRITICAL ANALYSIS GUIDELINES:*
#         - Examine contextual nuances.
#         - Analyze emotional implications of word choices.
#         - Consider both explicit statements and implied attitudes.
#         - Provide a comprehensive yet concise analysis.
#         """
        
#         # Send message and get response
#         response = self.chat.send_message(prompt)
#         return response.text  

from agents.base_agent import BaseStanceAgent
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import json
import os

class SentimentAgent(BaseStanceAgent):
    def __init__(self, model_path="stance_detection_model_with_diverse_negation_newmodel"):
        super().__init__(
            role="Sentiment Analysis Specialist",
            goal="Analyze text to determine stance toward specific topics using a fine-tuned BERT model",
            backstory="""
            As a highly nuanced linguistic expert, you specialize in decoding the stance and attitudinal undertones 
            of textual communication. Your unique ability lies in parsing complex language to extract stance 
            signals that reveal an author's underlying position toward a specific topic. You leverage a fine-tuned
            BERT model specifically trained for stance detection with augmented data handling various negation forms.
            """,
            model="gemini-2.0-flash-lite"
        )
        
        # Load the BERT model, tokenizer and label encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned BERT model, tokenizer and label encoder"""
        try:
            # Load configuration
            with open(os.path.join(self.model_path, "config.json"), "r") as f:
                self.config = json.load(f)
            
            # Load model
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # Load label encoder
            self.label_encoder = joblib.load(os.path.join(self.model_path, "label_encoder.pkl"))
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Available stance labels: {self.label_encoder.classes_}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, text, topic):
        """Preprocess the text and topic for model input"""
        # Encode the text and topic for BERT
        topic_tokens = self.tokenizer.encode(topic, add_special_tokens=False)
        max_length = self.config.get("max_length", 128)
        remaining_tokens = max_length - len(topic_tokens) - 2
        
        tweet_tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(0, remaining_tokens)
        )
        
        tokens = [self.tokenizer.cls_token_id] + tweet_tokens + topic_tokens + [self.tokenizer.sep_token_id]
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens))
        
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]
        
        return {
            'input_ids': torch.tensor([tokens], dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        }
    
    def sentiment_analysis(self, text, topic):
        """
        Perform sentiment analysis using the fine-tuned BERT model
        Returns a detailed JSON with stance information
        """
        # Preprocess the input
        inputs = self.preprocess_input(text, topic)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Decode the prediction
        stance = self.label_encoder.inverse_transform([prediction])[0]
        
        # Ensure stance is a string, not a float
        if not isinstance(stance, str):
            stance = str(stance)
        
        # Map stance to sentiment information
        sentiment_mapping = {
            'favor': 'Positive',
            'against': 'Negative',
            'none': 'Neutral'
        }
        
        # Extract confidence scores for all classes
        all_confidences = {}
        for i in range(len(self.label_encoder.classes_)):
            class_name = self.label_encoder.inverse_transform([i])[0]
            if not isinstance(class_name, str):
                class_name = str(class_name)
            all_confidences[class_name] = probabilities[0][i].item()
        
        # Create a comprehensive result
        result = {
            "stance": stance,
            "confidence": confidence * 100,  # Convert to percentage
            "sentiment_polarity": sentiment_mapping.get(stance, "Unknown"),
            "sentiment_intensity": min(10, confidence * 10),  # Scale 0-10
            "all_confidences": all_confidences,
            "input_text": text,
            "topic": topic
        }
        
        # Create a formatted response string
        response = f"""
        ## Stance Analysis Results
        
        **Text:** "{text}"
        **Topic:** {topic}
        
        **Detected Stance:** {stance.upper()}
        **Confidence:** {confidence*100:.2f}%
        **Sentiment Polarity:** {sentiment_mapping.get(stance, "Unknown")}
        **Sentiment Intensity:** {min(10, confidence*10):.1f}/10
        
        **Confidence Breakdown:**
        {', '.join([f"{k}: {v*100:.2f}%" for k, v in all_confidences.items()])}
        
        This analysis was performed using a fine-tuned BERT model specifically trained for stance detection.
        """
        
        return response
    
    def get_stance(self, text, topic):
        """
        Simple method to just get the stance without detailed analysis
        Returns: stance label (favor, against, none)
        """
        inputs = self.preprocess_input(text, topic)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        stance = self.label_encoder.inverse_transform([prediction])[0]
        
        # Ensure stance is a string, not a float
        if not isinstance(stance, str):
            stance = str(stance)
            
        return stance