from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")


def chat(userprompt):
    prompt = (
        f"System: "
        f"Agent responses will be truthful, helpful, and harmless.\n"
        f"User: {userprompt}\n"
        f"Agent: "
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.2)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


if __name__ == "__main__":
    llm = LLM()
    print(llm.generate("How do I make Ramen noodles?"))
