import languagemodels as lm
import time
import resource
import json


def mem_used_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print(f"Memory used before loading models: {mem_used_gb():.2f}GB")


print("\n# Completion Test\n")

print(f'{lm.complete("They ran until")=}')

print("\n# Chat Test\n")

print(
    lm.chat(
        """
System: Respond helpfully. It is Monday

User: What day is it?

Assistant:
"""
    )
)


print("\n# Instruction Tests\n")

tests = [
    ("What is the capital of France?", "Paris"),
    (
        "Lungs are used for respiration in mammals. Computers are machines that do not respirate. Would you expect computers to have lungs?",
        "No",
    ),
    ("A game uses a bat and ball. Is it baseball or soccer?", "Baseball"),
    ("Is grass green or blue?", "Green"),
    ("Classify as positive or negative: He smells gross", "Negative"),
    ("Does a car have more wheels than a bike?", "Yes"),
    (
        "I have 3 books then lose 1. How many books do I have. Think step by step to get the correct answer.",
        "2",
    ),
]

accuracy = 0


start = time.perf_counter_ns()

lm.do("Test first run time")

print(f"Initialization time: {(time.perf_counter_ns() - start) / 1e6:.0f}ms")

print(f"Memory used after running chat inference: {mem_used_gb():.2f}GB")

start = time.perf_counter_ns()
chars_generated = 0

for test in tests:
    response = lm.do(test[0])
    chars_generated += len(response)
    if test[1].lower() in response.lower():
        accuracy += 1 / len(tests)
    print(test[0], response)

print(
    f"Average inference time: {(time.perf_counter_ns() - start)/len(tests)/1e6:.0f}ms"
)

print(
    f"{(time.perf_counter_ns() - start)/chars_generated/1e6:.0f}ms per character generated"
)

print(f"Overall accuracy: {accuracy:.2f}")

print(f"Memory used after running inference: {mem_used_gb():.2f}GB")

print("\n# Embedding Tests\n")

planets = json.load(open("test/planets.json"))

# Make sure the model is loaded before testing
start = time.perf_counter_ns()
lm.docs.store("just initializing")
lm.docs.clear()
print(f"Model load time: {(time.perf_counter_ns() - start) / 1e6:.0f}ms")

start = time.perf_counter_ns()
for planet in planets:
    lm.docs.store(planet["content"], planet["name"])
ms = (time.perf_counter_ns() - start) / 1e6
print(
    f"Embedded {len(lm.docs.chunks)} chunks in {ms:.0f}ms ({ms/len(lm.docs.chunks):.0f}ms per chunk)"
)

start = time.perf_counter_ns()
print(lm.get_doc_context("Which planets have rings?"))
print(f"Search time: {(time.perf_counter_ns() - start) / 1e6:.0f}ms")
lm.docs.clear()

max_ram = lm.models.get_max_ram()
print(
    f"Memory used after all tests: {mem_used_gb():.2f}GB (must be under {max_ram:.2f}GB)"
)

# Confirm that we fit in max_ram after running all tests
assert mem_used_gb() < max_ram
