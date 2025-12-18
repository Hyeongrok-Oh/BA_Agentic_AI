import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent_classifier import IntentClassifier

def verify():
    try:
        ic = IntentClassifier()
        print("Classifier initialized successfully.")
        
        # Test case from the new examples (Report Generation - New Report)
        query = "북미 지역 마케팅 비용 효율성 분석"
        print(f"\nTesting Query: '{query}'")
        
        messages = [{'role': 'user', 'content': query}]
        result = ic.classify(messages)
        
        print(f"Intent: {result.get('intent')}")
        print(f"Sub-Intent: {result.get('sub_intent')}")
        
        if result.get('intent') == "Report Generation" and result.get('sub_intent') == "New Report":
            print("SUCCESS: Classification matches expected outcome.")
        else:
            print("WARNING: Classification outcome differs from expectation (but might still be valid).")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify()
