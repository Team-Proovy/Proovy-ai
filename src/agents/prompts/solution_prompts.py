from agents.prompts.loader import load_prompt, render_prompt


SOLUTION_JSON_EXAMPLE = load_prompt("solution/json_example.txt").strip()
SOLUTION_SYSTEM_PROMPT = load_prompt("solution/system.txt").strip()


def build_solution_user_prompt(problems_json: str) -> str:
    return render_prompt(
        "solution/user.txt",
        solution_json_example=SOLUTION_JSON_EXAMPLE,
        problems_json=problems_json,
    ).strip()
