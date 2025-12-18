"""
Domain Guardrail Layer for LG Electronics HE Division Data Analysis Agent

This module implements a pre-filtering guardrail layer that checks if a user query
is related to the business domain BEFORE passing it to the Intent Classifier.

Best Practices Applied:
- LLM-based domain classification (GPT-4o-mini)
- Fast rejection of non-business queries
- Separation of concerns (guardrail vs intent classification)
"""

import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class GuardrailResult(BaseModel):
    """Guardrail classification result"""
    is_business_related: bool = Field(
        ..., 
        description="True if the query is related to business data analysis, False otherwise"
    )
    category: Literal["Business", "Non-Business"] = Field(
        ..., 
        description="Query category"
    )
    non_business_type: Optional[Literal["Weather", "Food", "Entertainment", "General Chat", "Other"]] = Field(
        None, 
        description="Type of non-business query if applicable"
    )
    response_message: Optional[str] = Field(
        None, 
        description="Response message for non-business queries"
    )
    recommended_questions: Optional[List[str]] = Field(
        None, 
        description="Recommended business-related questions"
    )


class DomainGuardrail:
    """
    Domain Guardrail Layer
    
    Filters out non-business queries before intent classification.
    Uses GPT-4o-mini for fast, cost-effective classification.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key is required.")
        self.client = OpenAI(api_key=self.api_key)
    
    def check(self, user_query: str, context_entities: dict = None, recent_history: list = None) -> dict:
        """
        Check if the query is business-related.
        
        Args:
            user_query: The user's input query
            context_entities: Current extract entities (e.g., {'company': 'LGE'})
            recent_history: Recent conversation messages
            
        Returns:
            dict: GuardrailResult with is_business_related flag
        """
        
        # Format Context Info for the prompt
        context_str = "None"
        if context_entities:
            context_str = str(context_entities)
            
        history_str = "None"
        if recent_history:
            # Simple formatter for history
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        system_prompt = f"""You are a Domain Guardrail for an LG Electronics HE Division data analysis agent.
Your ONLY job is to determine if the user's query is related to business data analysis or NOT.

## Context Information
You must consider the conversation history and current context.
- **Current Active Context**: {context_str}
- **Recent Conversation History**:
{history_str}

## Business Domain (PASS - is_business_related: true)
The following topics are BUSINESS-RELATED and should PASS:
- Revenue, Sales, Profit, Cost analysis
- Market share, Rankings, Comparisons
- Product performance (TV, OLED, QNED, etc.)
- Regional/Channel analysis (North America, Europe, Best Buy, etc.)
- Financial reports, Strategic planning, Risk analysis
- Time-based queries (Q1-Q4, 2023-2024, monthly data)
- ANY query that mentions companies, products, revenue, profit, sales, cost, market, or data

## CRITICAL: Multi-turn Follow-up Rules
If the user's input is short or ambiguous (e.g., "2024년", "미국", "OLED") BUT the **Current Active Context** or **History** indicates a business topic was already established:
- **YOU MUST PASS IT.** (Treat it as a valid follow-up business query)
- Example: History ["LG전자 매출 알려줘"], Input ["2024년"] -> **PASS** (It means "2024년 LG전자 매출")

## Non-Business Domain (REJECT - is_business_related: false)
The following are NOT business-related and should be REJECTED **ONLY IF** there is no business context:
- **Weather**: "오늘 날씨 어때?", "내일 비 올까?"
- **Food**: "점심 뭐 먹을까?", "맛집 추천해줘"
- **Entertainment**: "영화 추천해줘", "게임 추천", "음악 틀어줘"
- **General Chat**: "안녕", "고마워", "잘자", "뭐해?", "심심해"
- **Other**: Jokes, personal advice, general knowledge unrelated to business

## Classification Rules
1. If the query mentions ANY business keyword (매출, 영업이익, 시장점유율, 분석, 보고서, etc.) → Business
2. If the query asks about companies, products, or financial data → Business
3. Greetings like "안녕하세요" at the START of a business query → Business (pass the whole query)
4. **[Multi-turn]** If input is a short entity (Year, Company, Product) and fits the previous business context → Business
5. Pure greetings/chat without business context → Non-Business

## Response for Non-Business
If Non-Business, provide:
- response_message: "죄송합니다. 저는 LG전자 HE사업부 비즈니스 데이터 분석에 특화된 에이전트입니다."
- recommended_questions: Suggest 4 example business questions
"""
        
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format=GuardrailResult,
                temperature=0.0  # Deterministic for consistent filtering
            )
            
            result = response.choices[0].message.parsed
            return result.model_dump()
            
        except Exception as e:
            # On error, default to PASS (don't block legitimate queries)
            print(f"Guardrail error: {e}")
            return {
                "is_business_related": True,
                "category": "Business",
                "non_business_type": None,
                "response_message": None,
                "recommended_questions": None
            }
