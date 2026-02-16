from agents.prompts.loader import load_prompt, render_prompt


REVIEW_SYSTEM_PROMPT = load_prompt("review/system.txt").strip()
SUGGESTION_SYSTEM_PROMPT = load_prompt("review/suggestion_system.txt").strip()


def build_review_user_prompt(
    *,
    reasons: str,
    last_user_msg: str,
    feature_summary: str,
) -> str:
    return render_prompt(
        "review/review_user.txt",
        reasons=reasons,
        last_user_msg=last_user_msg,
        feature_summary=feature_summary,
    ).strip()


def build_suggestion_user_prompt(
    *,
    review_state: str,
    last_user_msg: str,
    solve_progress: str,
    feature_summary: str,
) -> str:
    return render_prompt(
        "review/suggestion_user.txt",
        review_state=review_state,
        last_user_msg=last_user_msg,
        solve_progress=solve_progress,
        feature_summary=feature_summary,
    ).strip()
