"""
Semantic Validator: Validates user-requested metrics against available data
using fuzzy matching and semantic understanding.
"""

class SemanticValidator:
    def __init__(self):
        # Available metrics in the LGE HE ERP database
        self.available_metrics = [
            "매출", "매출액", "순매출", "Revenue", "Sales",
            "영업이익", "영업이익률", "Operating Profit", "Operating Margin",
            "판매량", "판매 수량", "Units Sold", "Sales Volume",
            "시장점유율", "Market Share",
            "재고", "재고량", "Inventory",
            "원가", "제조원가", "COGS", "Cost of Goods Sold",
            "물류비", "Logistics Cost",
            "관세", "Tariff",
            "가격", "단가", "Price", "Unit Price",
            "할인", "Discount",
            "고객수", "Customer Count"
        ]
        
        # Unavailable metrics (explicitly not in our database)
        self.unavailable_metrics = [
            "마케팅 예산", "Marketing Budget",
            "광고비", "Advertising Spend",
            "R&D 투자", "R&D Investment",
            "직원수", "Employee Count",
            "급여", "Salary",
            "경쟁사 내부 데이터"
        ]
    
    def validate_metrics(self, requested_metrics):
        """
        Validates if requested metrics are available in the database.
        
        Args:
            requested_metrics: List of metric names (could be in Korean or English)
        
        Returns:
            dict with keys: valid_metrics, invalid_metrics, unavailable_metrics, suggestions
        """
        result = {
            "valid_metrics": [],
            "invalid_metrics": [],
            "unavailable_metrics": [],
            "suggestions": []
        }
        
        for metric in requested_metrics:
            metric_lower = metric.lower().strip()
            
            # Check if explicitly unavailable
            if any(unavail.lower() in metric_lower for unavail in self.unavailable_metrics):
                result["unavailable_metrics"].append(metric)
                result["suggestions"].append(f"'{metric}'은(는) 제공되지 않는 데이터입니다.")
                continue
            
            # Check if available (fuzzy match)
            found = False
            for available in self.available_metrics:
                if available.lower() in metric_lower or metric_lower in available.lower():
                    result["valid_metrics"].append(metric)
                    found = True
                    break
            
            if not found:
                # Semantic matching (simple keyword-based for now)
                if any(keyword in metric_lower for keyword in ["이익", "profit", "수익"]):
                    result["valid_metrics"].append("영업이익")
                    result["suggestions"].append(f"'{metric}' → '영업이익'로 해석했습니다.")
                elif any(keyword in metric_lower for keyword in ["매출", "revenue", "sales", "돈"]):
                    result["valid_metrics"].append("매출액")
                    result["suggestions"].append(f"'{metric}' → '매출액'로 해석했습니다.")
                elif any(keyword in metric_lower for keyword in ["판매", "물량", "수량", "volume"]):
                    result["valid_metrics"].append("판매량")
                    result["suggestions"].append(f"'{metric}' → '판매량'로 해석했습니다.")
                else:
                    result["invalid_metrics"].append(metric)
                    result["suggestions"].append(f"'{metric}'을(를) 인식할 수 없습니다. 다시 확인해주세요.")
        
        return result

# Singleton instance
validator = SemanticValidator()
