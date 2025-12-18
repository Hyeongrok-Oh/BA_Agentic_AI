class AgentOrchestrator:
    def generate_report(self, entities):
        """
        Mock implementation of report generation structure.
        """
        return {
            "company": entities.get("company", "LGE"),
            "period": entities.get("period", {}),
            "sections_to_generate": ["financial_analysis", "segment_analysis"],
            "section_configs": {
                "financial_analysis": {"metrics": ["Revenue", "Profit"]},
                "segment_analysis": {"metrics": ["Sales by Region"]}
            }
        }

orchestrator = AgentOrchestrator()
