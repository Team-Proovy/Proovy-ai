from agents.prompts.loader import render_prompt


def build_simple_response_system_prompt(history_context: str) -> str:
    return render_prompt(
        "maingraph/simple_response_system.txt",
        history_context=history_context,
    ).strip()
