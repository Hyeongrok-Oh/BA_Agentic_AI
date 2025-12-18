"""SQL Agent 테스트"""

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv('/Users/hyeongrokoh/BI/.env')

from sql_agent import SQLAgent

def test_sql_agent():
    db_path = '/Users/hyeongrokoh/BI/sql/lge_he_erp.db'
    api_key = os.getenv('OPENAI_API_KEY')

    print(f'DB Path: {db_path}')
    print(f'API Key: {"설정됨" if api_key else "없음"}\n')

    agent = SQLAgent(db_path, api_key)

    # 테스트 케이스들
    test_queries = [
        # 1. 단순 매출 조회
        "LG전자 2024년 3분기 북미 매출 얼마야?",

        # 2. 원가 조회
        "2024년 3분기 물류비(LOG) 총액은?",

        # 3. 비교 분석
        "2023년 Q4와 2024년 Q4 북미 매출 비교해줘",
    ]

    print('=' * 60)
    print('SQL Agent 테스트')
    print('=' * 60)

    for i, query in enumerate(test_queries, 1):
        print(f'\n### Test {i}: {query}')
        print('-' * 50)

        result = agent.query(query)

        if result.get('error'):
            print(f'❌ Error: {result["error"]}')
        else:
            print(f'\n📊 Reasoning:\n{result["reasoning"][:300]}...\n')
            print(f'📝 SQL:\n{result["sql"]}\n')

            if result['data'] is not None and not result['data'].empty:
                print(f'📈 Result ({len(result["data"])} rows):')
                print(result['data'].to_string(index=False))
            else:
                print('⚠️ No data returned')

        print('=' * 60)


if __name__ == '__main__':
    test_sql_agent()
