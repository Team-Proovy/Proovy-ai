from agents.prompts.loader import load_prompt, render_prompt


SPLIT_SYSTEM_PROMPT = load_prompt("splitter/system.txt").strip()


def build_split_user_prompt(text: str) -> str:
    return render_prompt("splitter/user.txt", text=text).strip()
