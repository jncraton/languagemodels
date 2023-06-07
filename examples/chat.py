import languagemodels as lm

prompt = f"System: Reply as a helpful assistant. Currently {lm.get_date()}."

while True:
    user_message = input("\nUser: ")

    prompt += f"\n\nUser: {user_message}"

    print(prompt)

    prompt += "\n\nAssistant:"

    response = lm.chat(prompt)
    print(f"\nAssistant: {response}")

    prompt += f" {response}"
