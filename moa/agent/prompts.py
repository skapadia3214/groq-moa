SYSTEM_PROMPT = """\
You are a personal assistant that is helpful.

{helper_response}\
"""

REFERENCE_SYSTEM_PROMPT = """\
You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. 
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
{responses}
"""