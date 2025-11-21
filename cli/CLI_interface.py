import subprocess
import os
import colorama
from colorama import Fore, Style

colorama.init()

# EXACT NAME OF YOUR MODEL IN OLLAMA
MODEL_NAME = "psych-therapist"

def run_model(prompt):
    """Runs Ollama with your model and returns the clean output."""
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )

    return result.stdout.strip()


def header():
    print(Fore.CYAN + "╔══════════════════════════════════════════════════════╗")
    print("║                 psych-therapist — CLI THERAPIST      ║")
    print("║         Fine-tuned Model using TinyLlama + Ollama    ║")
    print("╚══════════════════════════════════════════════════════╝" + Style.RESET_ALL)
    print("\nType " + Fore.YELLOW + "'exit'" + Style.RESET_ALL + " to quit.\n")


def main():
    os.system("cls" if os.name == "nt" else "clear")
    header()

    while True:
        user_input = input(Fore.GREEN + "You ▶ " + Style.RESET_ALL)

        if user_input.lower().strip() in ["exit", "quit"]:
            print(Fore.MAGENTA + "\nExiting… take care" + Style.RESET_ALL)
            break

        
        prompt = f"<|user|>\n{user_input}\n<|assistant|>"

        print(Fore.BLUE + "psych-therapist is thinking..." + Style.RESET_ALL)
        response = run_model(prompt)

        print("\n" + Fore.YELLOW + "psych-therapist replies:" + Style.RESET_ALL)
        print(Fore.WHITE + response + Style.RESET_ALL)
        print(Fore.CYAN + "\n──────────────────────────────────────────────\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
