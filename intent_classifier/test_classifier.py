"""Intent Classifier 테스트"""

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv('/Users/hyeongrokoh/BI/.env')

from src.intent_classifier import IntentClassifier

def test_classifier():
    api_key = os.getenv('OPENAI_API_KEY')
    print(f'OpenAI API Key: {"설정됨" if api_key else "없음"}\n')

    if not api_key:
        print("API Key가 없습니다!")
        return

    classifier = IntentClassifier(api_key)

    # 테스트 케이스들
    test_cases = [
        # Descriptive (단순 조회)
        {'query': 'LG전자 2024년 3분기 매출 얼마야?', 'expected': 'Data QA / Descriptive'},
        # Diagnostic (인과 분석)
        {'query': '환율이 왜 매출에 영향을 줘?', 'expected': 'Data QA / Diagnostic'},
        # Report Generation
        {'query': 'LG전자 3분기 실적 보고서 만들어줘', 'expected': 'Report Generation'},
    ]

    print('=== Intent Classifier Test ===\n')

    for tc in test_cases:
        messages = [{'role': 'user', 'content': tc['query']}]
        result = classifier.classify(messages)

        print(f"Q: {tc['query']}")
        print(f"Expected: {tc['expected']}")
        print(f"Result: {result.get('intent')} / {result.get('analysis_mode')}")
        print(f"Sub-intent: {result.get('sub_intent')}")

        if result.get('clarifying_question'):
            print(f"Clarifying: {result.get('clarifying_question')}")

        if result.get('extracted_entities'):
            entities = result.get('extracted_entities')
            print(f"Entities: company={entities.get('company')}, period={entities.get('period')}")

        print('-' * 50)


if __name__ == '__main__':
    test_classifier()
