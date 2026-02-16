from agents.prompts.loader import load_prompt, render_prompt


ANALYSIS_SYSTEM_PROMPT = load_prompt("solve/analysis_system.txt").strip()
STRATEGY_SYSTEM_PROMPT = load_prompt("solve/strategy_system.txt").strip()
FINAL_SUMMARY_SYSTEM_PROMPT = load_prompt("solve/final_summary_system.txt").strip()
WRITER_SYSTEM_PROMPT = load_prompt("solve/writer_system.txt").strip()


def build_analysis_user_prompt(
    *,
    user_text: str,
    ocr_text: str,
    context_section: str,
    indexed_problem_number: str,
    indexed_problem_text: str,
) -> str:
    return render_prompt(
        "solve/analysis_user.txt",
        user_text=user_text,
        ocr_text=ocr_text,
        context_section=context_section,
        indexed_problem_number=indexed_problem_number,
        indexed_problem_text=indexed_problem_text,
    ).strip()


def build_strategy_user_prompt(analysis_payload: str) -> str:
    return render_prompt(
        "solve/strategy_user.txt",
        analysis_payload=analysis_payload,
    ).strip()


def build_writer_user_prompt(serialized_data: str) -> str:
    return render_prompt(
        "solve/writer_user.txt",
        serialized_data=serialized_data,
    ).strip()
