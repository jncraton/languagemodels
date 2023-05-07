import languagemodels

tests = [
    ("What is the capital of France?", "Paris"),
    ("Lungs are used for respiration in mammals. Do computers have lungs?", "No"),
    ("I am playing a sport with a bat and ball. Am I playing baseball or basketball?", "Baseball"),
    ("What color is the sun, yellow or blue?", "Yellow")
]

accuracy = 0

for test in tests:
    response = languagemodels.chat(test[0])
    if test[1] in response:
        accuracy += 1 / len(tests)
    print(test[0], response)

print(f"Overall accuracy: {accuracy}")
