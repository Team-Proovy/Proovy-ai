from agents.prompts.loader import load_prompt, render_prompt


COMPLEXITY_SYSTEM_PROMPT = load_prompt("router/complexity_system.txt").strip()
PLANNER_SYSTEM_PROMPT = load_prompt("router/planner_system.txt").strip()
STEM_CLASSIFIER_SYSTEM_PROMPT = load_prompt("router/stem_system.txt").strip()


def build_complexity_user_prompt(question: str) -> str:
    return render_prompt("router/complexity_user.txt", question=question).strip()


def build_primary_feature_system_prompt(allowed: str) -> str:
    return render_prompt(
        "router/primary_feature_system.txt",
        allowed=allowed,
    ).strip()


def build_primary_feature_user_prompt(question: str) -> str:
    return render_prompt("router/primary_feature_user.txt", question=question).strip()


def build_planner_question_section(question: str) -> str:
    return render_prompt("router/planner_question_section.txt", question=question).strip()


def build_planner_hints_section(hints: str) -> str:
    return render_prompt("router/planner_hints_section.txt", hints=hints).strip()


def build_planner_output_instruction(allowed: str) -> str:
    return render_prompt(
        "router/planner_output_instruction.txt",
        allowed=allowed,
    ).strip()


def build_stem_context_section(conversation_context: str) -> str:
    return render_prompt(
        "router/stem_context_section.txt",
        conversation_context=conversation_context,
    ).rstrip()


def build_stem_user_prompt(context_info: str, combined_question: str) -> str:
    return render_prompt(
        "router/stem_user.txt",
        context_info=context_info,
        combined_question=combined_question,
    ).strip()
