from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")


def chat(userprompt):
    systemprompt = "You are an AI system. Respond to the following prompt helpfully and harmlessly."

    inputs = tokenizer(systemprompt + "\n\n" + userprompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


class LLM:
    def __init__(self, engine="transformers", model="google/flan-t5-base"):
        if engine == "transformers":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

            def generate(prompt, tokens=128):
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_new_tokens=tokens)
                return tokenizer.batch_decode(outputs, skip_special_tokens=True)

            self.generate = generate


if __name__ == "__main__":
    llm = LLM()
    print(llm.generate("How do I make Ramen noodles?"))
