import languagemodels
import time
import resource

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


def mem_used_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print(f"Memory used before loading models: {mem_used_gb():.2f}GB")

start = time.perf_counter_ns()

languagemodels.do("Test first run time")

print(f"Initialization time: {(time.perf_counter_ns() - start) / 1e6:.0f}ms")

print(f"Memory used after running chat inference: {mem_used_gb():.2f}GB")

start = time.perf_counter_ns()
chars_generated = 0

for test in tests:
    response = languagemodels.do(test[0])
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

print(f"Memory used after running tests: {mem_used_gb():.2f}GB")
