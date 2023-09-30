import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    chatbot_output = model.generate(
        input_ids=input_ids,
        max_length=50,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )
    chatbot_response = tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
    return chatbot_response

def generate_response_for_web(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    chatbot_output = model.generate(
        input_ids=input_ids,
        max_length=50,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )
    chatbot_response = tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
    return chatbot_response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        chatbot_response = generate_response(user_input)
        print("Chatbot: " + chatbot_response)
