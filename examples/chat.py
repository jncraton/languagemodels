import languagemodels as lm

prompt = (
    f"<|system|>Currently {lm.get_date()}.\n"
    f"Assistant responses are true helpful and harmless.<|endoftext|>"
)

while True:
    user_message = input('User: ')

    prompt += f"<|prompter|>{user_message}<|endoftext|><|assistant|>"

    print(prompt)

    response = lm.chat(prompt)
    print(f"Assistant: {response}")

    prompt += f"{response}<|endoftext|>"
