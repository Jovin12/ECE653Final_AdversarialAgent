from llm_models import target_agent

def load_llm_target(model_name='qwen3.5:0.8b'):
    """Return the target LLM for attacks"""
    # The target is already loaded in llm_models.py
    return target_agent