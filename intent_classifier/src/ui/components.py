import streamlit as st
from db_schema import AVAILABLE_DATA_INFO

def render_agent_suggestions(result):
    """
    Render suggestions or insights from the agent.
    """
    if result.get("insight"):
        st.info(f"💡 **Insight**: {result['insight']}")
    
    if result.get("recommended_question"):
        st.markdown(f"👉 **추천 질문**: {result['recommended_question']}")

def render_non_business_response():
    """
    Case 1: Out-of-Domain (Non-Business)
    Display service explanation and intent-based sample questions.
    """
    # Service description
    st.warning("⚠️ 죄송합니다. 저는 비즈니스 데이터 분석에 특화된 에이전트입니다.")
    
    # Get service info from db_schema
    service_desc = AVAILABLE_DATA_INFO.get("service_description", "")
    st.markdown(service_desc)
    
    # Display intent-based sample questions
    st.markdown("### 💡 이런 질문을 해보세요!")
    
    sample_questions = AVAILABLE_DATA_INFO.get("sample_questions", {})
    
    # Report Generation examples
    st.markdown("**📊 보고서 생성 (Report Generation)**")
    for q in sample_questions.get("Report Generation", []):
        st.markdown(f"- {q}")
    
    # Data QA examples
    st.markdown("**📈 데이터 조회 (Data QA)**")
    for q in sample_questions.get("Data QA", []):
        st.markdown(f"- {q}")

def render_data_unavailable_response(result):
    """
    Case 3: Data Unavailable
    Display available date range and suggest alternative questions.
    Note: recommended_questions are displayed by app.py, not here (to avoid duplication).
    """
    # Use response_message (Korean), NOT thinking (English reasoning)
    response_msg = result.get("response_message", "요청하신 데이터는 제공 가능 기간을 벗어납니다.")
    st.error(f"⛔ **데이터 제공 불가**: {response_msg}")
    
    # Get available date range
    date_range = AVAILABLE_DATA_INFO.get("date_range", {})
    date_display = date_range.get("display", "정보 없음")
    
    st.markdown(f"""
### 📅 제공 가능한 데이터 기간
**{date_display}**

### 📋 확인 가능한 데이터
- **회사**: {AVAILABLE_DATA_INFO.get('company', 'N/A')}
- **지역**: {', '.join(AVAILABLE_DATA_INFO.get('regions', []))}
- **제품**: {', '.join(AVAILABLE_DATA_INFO.get('products', []))}
""")
    
    # Suggest alternative questions based on available data
    st.markdown("### 💡 이 질문은 어떠세요?")
    
    # Get recommended questions from result if available
    recommended = result.get("recommended_questions", [])
    if recommended:
        for q in recommended:
            st.markdown(f"- {q}")
    else:
        # Default fallback questions
        sample_questions = AVAILABLE_DATA_INFO.get("sample_questions", {})
        for q in sample_questions.get("Data QA", []):
            st.markdown(f"- {q}")

def render_missing_slot_response(clarifying_question, result):
    """
    Case 2: Missing Required Slots
    Display the clarifying question with helpful context.
    """
    st.warning(f"🤔 {clarifying_question}")
    
    # Show what we already know
    entities = result.get("extracted_entities", {})
    if entities:
        known_info = []
        if entities.get("company"):
            known_info.append(f"회사: {entities['company']}")
        if entities.get("region"):
            region = entities['region']
            if isinstance(region, list):
                known_info.append(f"지역: {', '.join(region)}")
            else:
                known_info.append(f"지역: {region}")
        if entities.get("product"):
            product = entities['product']
            if isinstance(product, list):
                known_info.append(f"제품: {', '.join(product)}")
            else:
                known_info.append(f"제품: {product}")
        
        if known_info:
            st.info(f"✅ 확인된 정보: {' | '.join(known_info)}")
