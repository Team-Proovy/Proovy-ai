from agents.prompts.loader import load_prompt


PARTIAL_RESPONSE_SYSTEM_PROMPT = load_prompt("final_response/partial_system.txt").strip()
PARTIAL_RESPONSE_USER_INSTRUCTION = load_prompt(
    "final_response/partial_instruction.txt"
).strip()
TRADITIONAL_RESPONSE_SYSTEM_PROMPT = load_prompt(
    "final_response/traditional_system.txt"
).strip()
TRADITIONAL_RESPONSE_USER_INSTRUCTION = load_prompt(
    "final_response/traditional_instruction.txt"
).strip()
