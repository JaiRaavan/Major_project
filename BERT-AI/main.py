import os
import traceback
from dotenv import load_dotenv
from agents.coordinator_agent import StanceCoordinatorAgent

def main():
    try:
        # Load environment variables
        load_dotenv()

        # Verify API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("ERROR: No Google API key found. Please check your .env file.")
            return

        # Initialize the Stance Coordinator Agent
        stance_coordinator = StanceCoordinatorAgent()

        # Example usage
        # text = "Ms Ramos don't adult women have the right to decide what they do or do not want to do ? Isn't it equally coercive to restrict their freedom ? I also question the realism of your argument. AFAIK, prostitution seems to be a major sector of every known human society, in societys with strict penaltys (even death) and in societys where it is legal Isn't crimminlization like prohibition - something doomed to failure, as banning prostitution is like banning alcohol, or other drugs, perhaps desirable, but doomed to fail because it goes against human nature ? I think you are also confusing social class with prostitution; much prostituion is unpleasant because it is around poverty, just as coal mining is unpleasant."
        text= "I strongly believe we need not to increase funding for public education."
        topic= "Education Funding"

        # Detect stance
        stance_result = stance_coordinator.synthesize_stance(text, topic)
        
        print("\nStance Detection Result:")
        print(stance_result)

    except Exception as e:
        print("An error occurred:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()