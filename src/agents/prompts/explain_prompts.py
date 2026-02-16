from agents.prompts.loader import load_prompt, render_prompt


EXPLAIN_CONTEXT_INFO_PROMPT = load_prompt("explain/context_info.txt").rstrip()


def build_explain_system_prompt(context_info: str) -> str:
    return render_prompt(
        "explain/system.txt",
        context_info=context_info,
    ).strip()


def build_explain_history_section(conversation_context: str) -> str:
    return render_prompt(
        "explain/history_section.txt",
        conversation_context=conversation_context,
    ).rstrip()


def build_explain_user_prompt(history_section: str, user_text: str) -> str:
    return render_prompt(
        "explain/user.txt",
        history_section=history_section,
        user_text=user_text,
    ).strip()
