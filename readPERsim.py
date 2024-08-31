import json
import random
import time
import sys
import signal
import re
from colorama import init, Fore, Style, Back

init(autoreset=True)  # Initialize colorama

def load_responses(file_path):
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if 'assistant_response' in entry:
                responses.append(entry['assistant_response'])
    return responses

def format_markdown(text):
    lines = text.split('\n')
    formatted_lines = []
    in_simthinking = False

    for line in lines:
        # SimThinking blocks
        if line.strip() == '<simthinking>':
            in_simthinking = True
            formatted_lines.append(Fore.CYAN + Style.BRIGHT + line + Style.RESET_ALL)
            continue
        elif line.strip() == '</simthinking>':
            in_simthinking = False
            formatted_lines.append(Fore.CYAN + Style.BRIGHT + line + Style.RESET_ALL)
            continue
        
        if in_simthinking:
            # Bullet points in SimThinking
            if line.strip().startswith('•'):
                formatted_lines.append(Fore.YELLOW + line + Style.RESET_ALL)
            else:
                formatted_lines.append(Fore.CYAN + line + Style.RESET_ALL)
        else:
            # Horizontal rules
            if re.match(r'^─+$', line.strip()):
                formatted_lines.append(Fore.MAGENTA + line + Style.RESET_ALL)
            # Statement/Action
            elif line.startswith('[Statement/Action:'):
                formatted_lines.append(Fore.GREEN + line + Style.RESET_ALL)
            else:
                # Regular text
                formatted_lines.append(line)

    return '\n'.join(formatted_lines)

def print_random_response(responses):
    if responses:
        response = random.choice(responses)
        formatted_response = format_markdown(response)
        print(formatted_response)
    else:
        print(Fore.RED + "No responses found in the file." + Style.RESET_ALL)

def main(file_path, interval):
    responses = load_responses(file_path)
    
    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(Fore.CYAN + "Printing random Sim responses. Press Ctrl+C to exit." + Style.RESET_ALL)
    
    while True:
        print_random_response(responses)
        print(Fore.CYAN + "─" * 40 + Style.RESET_ALL)  # Separator between responses
        time.sleep(interval)

if __name__ == "__main__":
    #file_path = "simulation_log.jsonl"  # Update this if your file has a different name
    file_path = "archive\simulation_log_2908249PM.jsonl"
    interval = 5  # Time in seconds between prints
    
    main(file_path, interval)