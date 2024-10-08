import os
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



import os
import subprocess
import time

# Ekranı temizle
subprocess.call("cls", shell=True)

# ANSI renk kodları
colors = [
    "\033[38;5;196m",  # Kırmızı
    "\033[38;5;202m",  # Turuncu
    "\033[38;5;226m",  # Sarı
    "\033[38;5;82m",   # Yeşil
    "\033[38;5;45m",   # Mor
    "\033[38;5;34m",   # Mavi
    "\033[38;5;51m"    # Cyan
]
reset_color = "\033[0m"  # Renk sıfırlama

# ASCII sanatı
ascii_art = [
    "__                    ___                ",
    "/ /   ____  ____ _____/ (_)___  ____ _    ",
    "/ /   / __ \\/ __ `/ __  / / __ \\/ __ `/    ",
    "/ /___/ /_/ / /_/ / /_/ / / / / / /_/ / _ _ ",
    "/_____\\/____/\\__,_/\\__,_/_/_/ /_/\\__, (_|_|_)",
    "                                /____/       "
]

# Ekranın genişliği
screen_width = 80

# Her satırı yavaşça renkli yazdır
for index, line in enumerate(ascii_art):
    # Renk seçimi için renk dizisinden bir renk al
    color = colors[index % len(colors)]
    # Satırı ortala
    centered_line = line.center(screen_width)
    print(" ", end='')  # Boşluğu ekle
    for char in centered_line:
        print(color + char, end='', flush=True)
        time.sleep(0.02)  # Her karakter için 0.05 saniye bekle
    print(reset_color)  # Satırın sonunda rengi sıfırla

# Mesajları yavaşça yazdır
welcome_message = "Welcome to the FaxterGPT chat!"
leave_message = "Say bot `leave` to leave"

# Mesajları ortala
centered_welcome = welcome_message.center(screen_width)
centered_leave = leave_message.center(screen_width)

# Yavaşça yazdır
for char in centered_welcome:
    print(char, end='', flush=True)
    time.sleep(0.05)  # Her karakter için 0.05 saniye bekle
print()  # Yeni satıra geç

for char in centered_leave:
    print(char, end='', flush=True)
    time.sleep(0.05)  # Her karakter için 0.05 saniye bekle
print("\n")  # Yeni satıra geç


# Load the model and tokenizer
model_name = "microsoft/DialoGPT-large"  # Using a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Clear the screen
subprocess.call("cls", shell=True)

# ANSI color codes for sunrise effect
sunrise_colors = [
    "\033[38;5;214m",  # Orange
    "\033[38;5;226m",  # Yellow
    "\033[38;5;197m",  # Pink
    "\033[0m"  # Reset
]

# Color codes for user input and bot response
user_input_color = "\033[38;5;50m"  # Green
bot_response_color = "\033[38;5;81m"  # Blue
welcome_color = "\033[38;5;78m"  # Cyan
instruction_color = "\033[38;5;160m"  # Magenta

# ASCII art
ascii_art = [
    "        ______           __            __________  ______",
    "       / ____/___ __  __/ /____  _____/ ____/ __ \\_  __/",
    "      / /_  / __ `/ |/_/ __/ _ \\/ ___/ / __/ /_/ / / /   ",
    "     / __/ / /_/ />  </ /_/  __/ /  / /_/ / ____/ / /    ",
    "    /_/    \\__,_/_/|_|\\__/\\___/_/   \\____/_/     /_/   "
]

# Space settings
ascii_spaces = " " * 20  # 20 spaces for ASCII art
message_spaces = " " * 30  # 30 spaces for messages

# Function to print text slowly
def slow_print(text, delay=0.05):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # Move to the next line

# Function to print ASCII art once
def print_ascii_once():
    subprocess.call("cls", shell=True)  # Clear the screen
    for i, line in enumerate(ascii_art):
        # Set colors for the sunrise effect
        color = sunrise_colors[i % len(sunrise_colors)]
        slow_print(color + ascii_spaces + line + "\033[0m")  # Add color and reset
    slow_print(welcome_color + "\n" + message_spaces + "Welcome to the FaxterGPT chat!\033[0m")  # Welcome message in cyan
    slow_print(instruction_color + message_spaces + "Say bot `leave` to leave\n\n\n\033[0m")  # Instructions in magenta

# Print ASCII art at the start
print_ascii_once()

# Chat history
chat_history_ids = None

while True:
    user_input = input(user_input_color + "You: \033[0m")  # User input in green
    if user_input.lower() == "leave":
        slow_print("\n" + bot_response_color + "Bot: Goodbye!\033[0m")  # Leave message in blue
        break

    # Convert user input to tokens
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Update history
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    # Get response from the model
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # For response diversity
        top_k=50,  # Use the best 50 words for response generation
        top_p=0.95,  # Increase diversity
        do_sample=True  # Sample to create responses
    )

    # Decode the model response
    bot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Clear the screen before showing the bot response
    subprocess.call("cls", shell=True)
    slow_print(bot_response_color + f"Bot: {bot_response}\033[0m")  # Bot response in blue
