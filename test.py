import languagemodels

tests = [
    ("What is the capital of France?", "Paris"),
    ("I am a robot. Do I breath?", "No"),
    ("What is 2+2?", "4"),
    ("I am playing a sport with a bat and ball. What is the sport?", "Baseball"),
]

accuracy = 0

for test in tests:
    response = languagemodels.chat(test[0])
    if test[1] in response:
        accuracy += 1/len(tests)
    print(test[0], response)

print(f"Overall accuracy: {accuracy}")
