"""
Embedding Validator: Mock implementation for compatibility
"""

class EmbeddingValidator:
    def __init__(self):
        pass
    
    def validate(self, query):
        return {"status": "mock_implementation"}

def get_embedding_validator():
    return EmbeddingValidator()
