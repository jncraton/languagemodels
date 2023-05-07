import languagemodels

tests = [
    ("What is the capital of France?", "Paris"),
    ("Lungs are used for respiration in mammals. Do computers have lungs?", "No"),
    ("A game uses a bat and ball. Is it baseball or soccer?", "Baseball"),
    ("What color is the sun, yellow or blue?", "Yellow"),
]

accuracy = 0

for test in tests:
    response = languagemodels.chat(test[0])
    if test[1] in response:
        accuracy += 1 / len(tests)
    print(test[0], response)

print(f"Overall accuracy: {accuracy}")

assert accuracy >= 0.75
