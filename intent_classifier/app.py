import streamlit as st
import json
from src.intent_classifier import IntentClassifier
from src.agent_orchestrator import orchestrator
from src.guardrail import DomainGuardrail

# Page Config
st.set_page_config(
    page_title="의도 분류 에이전트",
    page_icon="🤖",
    layout="centered"
)

# Title and Description
st.title("🤖 의도 분류 에이전트")
st.markdown("""
이 에이전트는 사용자의 질문을 분석하여 **보고서 생성** 또는 **데이터 QA**인지 파악합니다.
**전자제품 기업(삼성전자, LG전자 등)** 관련 질문만 처리할 수 있습니다.
""")

# Sidebar for API Key
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key", type="password", help="OpenAI API Key를 입력하세요.")
    if not api_key:
        st.warning("API Key를 입력해야 사용할 수 있습니다.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 전자제품 기업 데이터 분석 도우미입니다. 무엇을 도와드릴까요?"}
    ]

# Initialize Context Entities (Phase 11: Multi-turn Memory)
if "context_entities" not in st.session_state:
    st.session_state.context_entities = {}

# Initialize Conversation State (Task Flow Control)
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "IN_PROGRESS"

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
# Chat Input Handling (Enhanced for Recommendation Chips)
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

user_input = st.chat_input("질문을 입력하세요 (예: 삼성전자 3분기 영업이익 보고서 만들어줘)")

# Check if triggered by button or input
prompt = user_input or st.session_state.pending_prompt

if prompt:
    # Reset pending prompt if used
    if st.session_state.pending_prompt == prompt:
        st.session_state.pending_prompt = None
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check API Key
    if not api_key:
        st.error("⚠️ 사이드바에서 API Key를 먼저 입력해주세요.")
    else:
        # Process Intent
        with st.chat_message("assistant"):
            
            # ============================================================
            # 🛡️ GUARDRAIL LAYER: Pre-filter Non-Business Queries
            # ============================================================
            with st.spinner("쿼리 유효성 검사 중..."):
                guardrail = DomainGuardrail(api_key)
                
                # [Multi-Turn Fix] Pass History & Context to Guardrail
                # 1. Get recent history (last 2 turns + current prompt effectively)
                recent_history = st.session_state.messages[-3:] if len(st.session_state.messages) > 0 else []
                
                # 2. Get active context entities
                active_context = st.session_state.context_entities
                
                guard_result = guardrail.check(prompt, context_entities=active_context, recent_history=recent_history)
                
            if not guard_result.get("is_business_related", True):
                # -------------------------------------------------------------
                # 💡 SMART GUIDE: Contextual Fallback for Non-Business Queries
                # -------------------------------------------------------------
                
                # 1. Friendly Guidance Message
                st.info("😅 죄송합니다. 일상적인 대화보다는 **비즈니스 데이터 분석**에 도움을 드릴 수 있습니다.")
                
                # 2. Recommended Question Chips
                recommendations = guard_result.get("recommended_questions", [])
                if recommendations:
                    st.markdown("### 💡 이런 질문은 어떠세요?")
                    cols = st.columns(2)
                    for idx, q in enumerate(recommendations[:4]): # Max 4 items
                        if cols[idx % 2].button(q, key=f"rec_{idx}"):
                            # Programmatically trigger the prompt
                            st.session_state.pending_prompt = q
                            st.rerun()
                
                # 3. Service Scope Expander
                with st.expander("📘 제공 가능한 데이터 범위 확인하기"):
                    st.markdown("""
                    *   **기업**: LG전자 HE본부 및 주요 경쟁사 (Samsung, Sony 등)
                    *   **주요 지표**: 매출액(Revenue), 영업이익(Profit), 판매량(Sales), 시장점유율(M/S)
                    *   **분석 기간**: 2023년 1분기 ~ 2024년 현재 (분기/월별)
                    *   **기능**: 실적 보고서 생성, 특정 데이터 조회(Data QA), 경쟁사 비교 분석
                    """)
                
                # Add persistence message to history (for next turn view)
                st.session_state.messages.append({"role": "assistant", "content": "비즈니스 데이터에 대해 질문해주시면 자세히 답변 드리겠습니다! (위의 추천 질문을 클릭해보세요)"})
                st.stop()  # Stop further processing
            
            # ============================================================
            # 🔍 INTENT CLASSIFICATION (Only for Business Queries)
            # ============================================================
            with st.spinner("의도를 분석 중입니다..."):
                classifier = IntentClassifier(api_key)

                # 🔧 Enhanced Multi-turn Conversation Management
                MAX_CONVERSATION_TURNS = 10
                recent_messages = st.session_state.messages[-MAX_CONVERSATION_TURNS:]
                valid_messages = []
                for msg in recent_messages:
                    content = msg.get("content")
                    if content and isinstance(content, str) and content.strip():
                        valid_messages.append(msg)
                
                if not valid_messages:
                    st.error("⚠️ 유효한 메시지가 없습니다. 다시 시도해주세요.")
                    st.stop()
                
                result = classifier.classify(valid_messages)
                
                # --- PHASE 11: CONTEXT MERGING & SMART CHECK (Enhanced with Context Reset) ---
                
                # 0. Topic Shift Detection
                continuity = result.get("context_continuity", "continue")
                changed_entities = result.get("changed_entities", [])
                
                if continuity == "new_topic":
                    st.session_state.context_entities = {}
                    print("[RESET] New Topic Detected: Context Reset")
                    
                elif continuity == "partial_change" and changed_entities:
                    for entity in changed_entities:
                        st.session_state.context_entities.pop(entity, None)
                    print(f"[PARTIAL] Partial Change Detected: Removed {changed_entities} from context")
                
                # [NEW] Implicit Context Reset Check (Strategy: context_reset_recommendation.md)
                # If Task was COMPLETED in previous turn, check if user is starting a fresh topic implicitly
                last_state = st.session_state.conversation_state
                new_entities = result.get("extracted_entities", {}) or {}
                
                if last_state == "COMPLETED" and continuity != "new_topic":
                    # Check if key entities (Company/Product) are present in NEW query
                    has_new_company = bool(new_entities.get("company"))
                    
                    # If Company is MISSING in new query (meaning it relies on old context)
                    if not has_new_company:
                        # Safety Stop! Ask for confirmation
                        prev_company = st.session_state.context_entities.get("company", "이전 회사")
                        
                        clarifying_q = f"이전과 동일하게 **{prev_company}**의 데이터를 원하시나요, 아니면 다른 회사를 원하시나요?"
                        
                        # Set result to Ambiguous behavior
                        result["intent"] = "Ambiguous"
                        result["clarifying_question"] = clarifying_q
                        result["extracted_entities"] = None # Don't merge yet
                        
                        # Reset continuity to prevent auto-merge below
                        continuity = "new_topic" 
                        print(f"[STOP] Context Reset Triggered: Checking implicit follow-up for {prev_company}")

                # 1. Merge Context
                if continuity != "new_topic" and new_entities:
                    for k, v in new_entities.items():
                        if v:
                            st.session_state.context_entities[k] = v
                
                # 2. Smart Check for Clarification
                clarifying_q = result.get("clarifying_question")
                if clarifying_q:
                    ctx = st.session_state.context_entities
                    has_company = bool(ctx.get("company"))
                    period = ctx.get("period", {}) or {}
                    has_year = bool(period.get("year"))
                    
                    has_period = True if has_year else False
                    
                    if has_company and has_period:
                        print("[SMART] Smart Check: Context has all info. Overriding clarification.")
                        clarifying_q = None
                        result["clarifying_question"] = None
                        result["extracted_entities"] = st.session_state.context_entities
                        
                        # Re-generate report structure if needed
                        if result.get("intent") in ["Report Generation", "Data QA"]:
                             entities = st.session_state.context_entities
                             report_structure = orchestrator.generate_report(entities)
                             if "section_configs" in report_structure:
                                 for section_key, config in report_structure["section_configs"].items():
                                     config.pop("data_sources", None)
                             result["report_structure"] = report_structure 

                # -----------------------------------------------

                if "error" in result:
                    response_text = f"오류가 발생했습니다: {result['error']}"
                    st.error(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                else:
                    intent = result.get("intent")
                    
                    # Extract UI-specific fields
                    clarifying_q = result.get("clarifying_question")
                    recommended_q = result.get("recommended_question")
                    insight = result.get("insight")
                    
                    # Display suggestions & responses
                    from src.ui.components import render_agent_suggestions, render_non_business_response, render_data_unavailable_response, render_missing_slot_response
                    render_agent_suggestions(result)
                    
                    # Handler Logic
                    if intent == "Out-of-Scope":
                        sub_intent_value = result.get("sub_intent")
                        if sub_intent_value == "Non-Business":
                            render_non_business_response()
                        elif sub_intent_value == "Data Unavailable":
                            render_data_unavailable_response(result)
                        
                        response_msg = result.get("response_message", "")
                        st.session_state.messages.append({"role": "assistant", "content": response_msg})
                        st.session_state.conversation_state = "COMPLETED" # End of turn
                    
                    elif result.get("sub_intent") == "Data Unavailable":
                        render_data_unavailable_response(result)
                        response_msg = result.get("response_message", "") or "데이터가 존재하지 않습니다."
                        st.session_state.messages.append({"role": "assistant", "content": response_msg})
                        st.session_state.conversation_state = "COMPLETED"

                    elif clarifying_q:
                        render_missing_slot_response(clarifying_q, result)
                        st.session_state.messages.append({"role": "assistant", "content": clarifying_q})
                        st.session_state.conversation_state = "IN_PROGRESS" # Still waiting

                    elif not clarifying_q:
                        # Success Case
                        sub_intent = result.get("sub_intent")
                        analysis_mode = result.get("analysis_mode")
                        detail_type = result.get("detail_type")
                        extracted_entities = result.get("extracted_entities")
                        report_structure = result.get("report_structure")
                        
                        success_msg = f"✅ **{intent}** 의도로 파악되었습니다."
                        if sub_intent: success_msg += f"\n- **유형**: {sub_intent}"
                        if analysis_mode: success_msg += f"\n- **분석 깊이**: {analysis_mode}"
                        if detail_type: success_msg += f"\n- **세부 유형**: {detail_type}"
                            
                        st.markdown(success_msg)
                        
                        # Display Report Structure
                        if report_structure:
                            st.success("📋 **다음 에이전트로 전달될 구조화된 데이터**")
                            # (Display logic simplified for brevity - kept logic same as original but cleaner)
                            if "company" in report_structure: st.write(f"🏢 **회사**: {report_structure.get('company')}")
                            if "period" in report_structure: st.write(f"📅 **기간**: {report_structure['period']}")
                            if "sections_to_generate" in report_structure:
                                st.write(f"📊 **생성될 섹션**: {len(report_structure['sections_to_generate'])}개")
                                with st.expander("📑 섹션 상세 보기"):
                                    st.json(report_structure['sections_to_generate'])
                            st.divider()
                        
                        elif extracted_entities:
                            st.success("📋 **다음 에이전트로 전달될 구조화된 데이터**")
                            st.json(extracted_entities)
                            st.divider()
                        
                        # Developer view
                        with st.expander("🔧 개발자용: 전체 JSON 보기"):
                            clean_result = {k: v for k, v in result.items() if k not in ["clarifying_question", "response_message", "report_structure"]}
                            st.json(clean_result)
                        
                        # Save & Message
                        output_file = "intent_output.json"
                        try:
                            save_data = {k: v for k, v in result.items() if k not in ["clarifying_question", "response_message"]}
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(save_data, f, indent=4, ensure_ascii=False)
                            
                            save_msg = f"결과가 `{output_file}`에 저장되었습니다. 다음 에이전트로 전달됩니다."
                            st.info(save_msg)
                            
                            full_response = f"{success_msg}\n\n{save_msg}"
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                            st.session_state.conversation_state = "COMPLETED" # Task Done!
                            
                        except Exception as e:
                            st.error(f"파일 저장 실패: {e}")

# Footer
st.markdown("---")
st.caption("Powered by OpenAI & Streamlit")
