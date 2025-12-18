import os
import json
from openai import OpenAI
from db_schema import DB_SCHEMA_PROMPT
from src.utils.example_selector import ExampleSelector  # [NEW]

class IntentClassifier:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key is required.")
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize Dynamic Example Selector
        try:
            # Path relative to this file: ./data/few_shot_examples.json
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "few_shot_examples.json")
            self.example_selector = ExampleSelector(data_path)
            print("Combine: Dynamic Few-Shot System Initialized")
        except Exception as e:
            print(f"Combine: Wrapper Init Failed {e}")
            self.example_selector = None

    def classify(self, messages):
        """
        Classifies the user's intent based on the conversation history.
        Injects the DB Schema to check for data availability.
        Uses OpenAI Structured Outputs for reliable JSON generation.
    - Implements Dynamic Few-Shot Prompting to retrieve relevant examples.
        """
        from src.schemas import IntentResult

        # --- Dynamic Few-Shot Injection ---
        user_query = messages[-1]['content'] if messages else ""
        dynamic_examples = self.example_selector.find_diverse_examples(user_query, k=3, ensure_category_diversity=True)
        
        dynamic_examples_text = ""
        for i, ex in enumerate(dynamic_examples, 1):
            dynamic_examples_text += f"\n**Dynamic Example {i} ({ex['category']})**\n"
            dynamic_examples_text += f"User: \"{ex['question']}\"\n"
            dynamic_examples_text += f"Output:\n{json.dumps(ex['answer'], ensure_ascii=False, indent=2)}\n"
        # ----------------------------------

        system_prompt = f"""You are an intelligent Intent Classifier for an LG Electronics HE Division data analysis agent.
Your job is to analyze the user's request and determine the intent, extracted entities, and whether the data exists in our database.

## 🛡️ [SAFETY & CONTEXT RULES] (CRITICAL)
1. **Strict Persistence Policy**: 
   - IF the current query implies missing slots (e.g. "What about 2024?" or "Compare with Samsung"):
     - YOU MUST carry over the **missing** company/product/metric from the previous conversation history.
   - Example: History(LGE Revenue) -> User("Samsung") -> Result(Samsung Revenue).
   - Example: History(LGE Revenue) -> User("What about 2023?") -> Result(LGE Revenue 2023).
   
2. **Explicit Overrides**: 
   - Information explicitly stated in the CURRENT query ALWAYS overrides the history.
   
3. **NO HALLUCINATION (Golden Rule)**: 
   - NEVER infer entities that were NOT mentioned in the current query OR the history.
   - If you don't know the company, DO NOT guess "LG Electronics" unless context supports it.
   
4. **Ambiguity Handling**: 
   - If the reference is ambiguous (e.g. "compare them" with no specific targets in history), output `Intent: Ambiguous` and ask for clarification.
   - Ambiguous words: "저거", "그거", "이전꺼" (Resolve these using History if possible, else ask).

{DB_SCHEMA_PROMPT}

## Hierarchical Intent Classification Rules (2D: Action x Depth)

### Dimension 1: Action (Intent & Sub-Intent)
**Level 1: Intent**
1. **Ambiguous**: Action unclear (Report or Data?).
2. **Report Generation**: Complex analysis or document request.
3. **Data QA**: Specific data points or insights.
4. **Out-of-Scope**: Unavailable data.

**Level 2: Sub-Intent (Data Source)**
- **Internal Data**: Internal DB metrics (Revenue, Profit, Sales).
- **External Data**: Competitor info, Market trends, News.
- **Hybrid Data**: Comparison (LGE vs Samsung), Internal + External factors.
- **Defined Report**: Standard periodic report.
- **New Report**: Custom ad-hoc report based on specific metric drivers.

### Dimension 2: Analysis Depth (Analysis Mode) - CRITICAL
This determines the workflow: Simple Retrieval vs Deep Analysis.

**1. Descriptive (Simple Retrieval)**
- **Goal**: "What", "When", "How much" (Fact checking).
- **Trigger**: User asks for values, trends, or simple status.
- **Examples**: 
  - "2024년 4분기 매출 얼마야?"
  - "북미 지역 판매량 알려줘"
  - "삼성전자 주가 조회해줘"
- **Routing**: Internal -> SQL Agent, External -> Search.

**2. Diagnostic (Deep/Causal Analysis)**
- **Goal**: "Why", "Cause", "Impact", "Reasoning" (In-depth investigation).
- **Trigger**: User asks about **reasons**, **causes**, **impacts**, or **analysis**.
- **Keywords**: "왜(Why)", "원인(Cause)", "이유(Reason)", "분석해줘(Analyze)", "영향(Impact)", "진단", "배경"
- **Examples**:
  - "매출이 **왜** 떨어졌어?" (Internal Data + Diagnostic)
  - "경쟁사 대비 부진한 **이유**가 뭐야?" (External Data + Diagnostic)
  - "환율이 영업이익에 미친 **영향** 분석해줘" (Hybrid Data + Diagnostic)
- **Routing**: Deep Analysis Pipeline (Hypothesis -> SQL -> GraphRAG).

### 3. Detail Type (Level 3)
**For Defined Report:**
- **Pre-closing**: Data for current/ongoing period (current month or quarter not yet finalized)
  - Keywords: "예상", "잠정", "현재 분기", "이번 달"
- **Post-closing**: Finalized historical data (past completed periods)
  - Keywords: Past quarters (Q1-Q3 2024), past years
- **External Event**: Analysis of external factor impacts
  - Keywords: "환율", "물류비", "관세", "원자재", "운송비", "Red Sea", "서플라이체인"

**For New Report:**
- detail_type: null (no predefined format)

**For Out-of-Scope (Data Unavailable):**
- **Required Slot Missing**: company 또는 period가 누락된 경우 (필수 슬롯 - 조회 불가)
- **Metric Unavailable**: 요청한 지표가 시스템에서 지원되지 않는 경우
- **Date Out of Range**: 데이터는 있지만 요청 기간(예: 1990년)에 해당 데이터 없음 (2023-2024만 지원)
- (Non-Business types are handled by Guardrail Layer)


## Multi-turn Response Generation Rules (CRITICAL)
Based on the intent classification, you MUST populate `response_message` and `recommended_questions`.

**Case 1: Out-of-Scope (Data Unavailable - unsupported company or metric)**
- Set `sub_intent`: "Data Unavailable"
- Set `response_message`: "죄송합니다. 현재 LG전자 HE사업부 데이터만 제공됩니다."
- Set `recommended_questions`: ["2024년 3분기 북미 OLED 수익성 분석 보고서 만들어줘", "2024년 3분기 북미 매출액 알려줘"]

**Case 2: Data Unavailable (Date out of range: before 2023 or after 2024)**
- Set `sub_intent`: "Data Unavailable"
- Set `response_message`: "요청하신 기간(예: 1990년)의 데이터는 제공되지 않습니다. 2023년~2024년 데이터만 확인 가능합니다."
- Set `recommended_questions`: Suggest similar queries with valid dates:
  - If user asked for "1990년 3분기 매출" → ["2024년 3분기 매출은 얼마인가요?", "2023년 3분기 매출과 비교해 보시겠어요?"]

**Case 3: Missing Essential Slots**
- Set `clarifying_question`: Ask for the missing slot in Korean.
- Do NOT set `recommended_questions` (user needs to answer first).


## Multi-turn Context Logic (Enhanced Rules)
Analyze the conversation history and determine context continuity based on these PRIORITY RULES:

**RULE 0 (HIGHEST PRIORITY): Detect Slot-Filling Responses**
→ If the PREVIOUS assistant message contains a `clarifying_question` (asking for company, period, etc.),
   and the user's CURRENT message is a SHORT answer (1-3 words) providing that slot:
   - Examples: "삼성전자", "LG전자", "2024년 3분기", "북미", "OLED"
   - This is a SLOT-FILLING response, NOT a new topic!
   - Set `context_continuity`: "continue"
   - Extract the entity from user's answer and MERGE with previous context
   - Logic for Competitors vs Unsupported:
     - If entity is **Competitor** (Samsung, Sony, etc.):
       - Set `intent`: "Data QA", `sub_intent`: "External Data" (Competitor analysis is Data QA, unless explicitly asked as Report)
     - If entity is truly **Unsupported/Irrelevant** (e.g., "Hyundai Motor"):
       - Set `intent`: "Out-of-Scope", `sub_intent`: "Data Unavailable"
       - Set `response_message`: "죄송합니다. 해당 데이터는 제공되지 않습니다."
     - **For Defined Reports**: If history has "Defined Report" intent and user provides Period/Company:
       - **EXCEPTION**: If the provided period is **Data Unavailable** (e.g. 2022, 1990), CLASSIFY as **Out-of-Scope**. Do NOT persist "Defined Report".
       - Otherwise: Set `intent`: "Report Generation", `sub_intent`: "Defined Report" (Maintain original intent)
     - Set `recommended_questions`: with LGE examples

**RULE 1: Check for "new_topic"**
→ If ALL of the following are true, classify as "new_topic":
  a) Previous message was NOT a clarifying question
  b) NO explicit reference to previous context ("it", "that", "그거", "거기서", "도" (also), "역시", "또한" etc.)
  c) At least 3+ entity types completely different from previous turn
  d) The query intent/domain is fundamentally different (e.g., Sales → Cost Reduction Strategy)

**RULE 2: Check for "partial_change" SECOND**
→ If user explicitly changes ONE existing entity value while keeping others:
  Examples:
  - "유럽은?" (changing region: NA → Europe)
  - "QNED는?" (changing product: OLED → QNED)
  - "3분기는?" (changing period: Q2 → Q3)
  - "2023년 10월도 가능해?" (changing period: Q3 → 2023-10, with "도" implying context continuation)
  
  CRITICAL: Changing an entity VALUE (e.g., Q3→Q4, NA→EU) = "partial_change"
  Even if time periods are consecutive (Q3→Q4), it's still a CHANGE, not continuation.
  
  Mark changed_entities: ["period"] or ["region"] or ["product"] etc.

**RULE 3: Default to "continue"**
→ If neither Rule 1 nor Rule 2 applies:
  - Adding NEW entities (e.g., adding "customer: Best Buy" when it wasn't mentioned before)
  - Adding NEW metrics (e.g., "영업이익도" when only Revenue was asked)
  - Refining/drilling down within the same context

**Context Continuity Decision Tree**:
```
Is it a completely different topic? (Rule 1) → YES → "new_topic"
  ↓ NO
Is user changing an existing entity value? (Rule 2) → YES → "partial_change"
  ↓ NO
Is user adding new info or drilling down? → YES → "continue"
```

## Company Extraction Rules
- When user explicitly says "LG전자", "LG Electronics", "LG", "우리", "엘지" → Extract as **"LGE"**
- When user mentions customer names (Best Buy, Amazon, Costco) in revenue queries → company: **"LGE"** (LGE's business with that customer)
- **If NO company is mentioned** → Set company to **null** and ask clarifying_question:
  - "어떤 회사의 데이터를 원하시나요? (현재 LG전자 HE사업부 데이터만 제공됩니다)"
  - NOTE: 향후 다양한 회사 데이터가 추가될 수 있으므로 기본값 적용하지 않음

## Product Extraction Rules
- **Single product**: Use string → "OLED"
- **Multiple products**: Use array → ["OLED", "QNED"]
- Preserve size mentions: "OLED 65인치" → "OLED 65"
- Multi-size comparisons → Array: "55인치 vs 65인치" → ["OLED 55", "OLED 65"]
- Size qualifiers: "65인치 이상" → "OLED 65+", "대형" → "OLED Large"
- Multiple products with slash: "OLED/QNED" → ["OLED", "QNED"]
- Product segments: "프리미엄" → "Premium TV"

## Region Extraction Rules
- **Single region**: Use string → "North America"
- **Multiple regions**: Use array → ["North America", "Europe"]
- Region comparisons: "북미 vs 유럽" → ["North America", "Europe"]

## Period Extraction Rules
Extract time periods with maximum granularity. Support ALL levels:
- **Year only**: {{"year": 2024, "quarter": null, "month": null, "day": null}}
- **Year + Quarter**: {{"year": 2024, "quarter": 3, "month": null, "day": null}}
- **Year + Single Month**: {{"year": 2024, "quarter": null, "month": 10, "day": null}} (single integer)
- **Year + Multiple Months**: {{"year": 2024, "quarter": null, "month": [7,8,9], "day": null}} (array for breakdowns)
- **Year + Month + Day**: {{"year": 2024, "quarter": null, "month": 10, "day": 15}}
- **Period Ranges**: Use month array for half-year/multi-quarter
  - "2024년 하반기" → {{"year": 2024, "quarter": null, "month": [7,8,9,10,11,12], "day": null}}
  - "2024년 3분기" (when asking for monthly breakdown) → {{"year": 2024, "quarter": 3, "month": [7,8,9], "day": null}}

## Clarification Rules (CRITICAL)

**Ambiguous Intent (Highest Priority):**
- If intent is "Ambiguous":
- Set `clarifying_question`: "Are you looking for a **Report** about [Entity], or specific **Data**?" (Translate to Korean: "[Entity]에 대해 보고서를 작성해 드릴까요, 아니면 특정 데이터를 조회해 드릴까요?")
- Leave `recommended_questions` as null.

If the user's request is for "Report Generation" or "Data QA" but is missing ESSENTIAL slots, you MUST ask a `clarifying_question`.

**Essential Slots (Required for ALL intents):**
1. **Company**: If NOT mentioned → Ask "어떤 회사의 데이터를 원하시나요? (현재 LG전자 HE사업부 데이터만 제공됩니다)"
2. **Period**: If NOT mentioned → Ask "어떤 기간의 데이터를 원하시나요?"

**Clarification Priority:**
- If BOTH company AND period are missing → Ask for company first
- If only period is missing → Ask for period

**Output Behavior:**
- If missing required slot(s): Set `clarifying_question` to the question string (in Korean).
- If all slots present: Set `clarifying_question` to null.

## Dynamic Few-Shot Examples (Context-Aware)
Using vector search to find the most relevant examples for your request:

{dynamic_examples_text}
"""


        # Prepare messages for API
        api_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            # Use beta.chat.completions.parse for Structured Outputs
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=api_messages,
                temperature=0,
                response_format=IntentResult
            )
            
            # The parsed result is already a Pydantic model
            result_obj = completion.choices[0].message.parsed
            
            # [Debugging] Print thinking process
            print("\n[Intent Classifier] Thinking Process:")
            if hasattr(result_obj.extracted_entities, 'thinking'):
                print(f"[Thinking] {result_obj.extracted_entities.thinking}")
            
            # Convert Pydantic model to dict for compatibility with existing app logic
            result = result_obj.model_dump()
            
            # Fix for "new_topic" bug (Legacy logic, kept for safety)
            if result.get("intent") == "new_topic":
                last_user_msg = messages[-1]['content'] if messages else ""
                if any(k in last_user_msg for k in ["순위", "랭킹", "Top", "top", "list", "목록"]):
                    result["intent"] = "Data QA"
                else:
                    result["intent"] = "Report Generation"
            
            return result

        except Exception as e:
            return {"error": str(e)}
