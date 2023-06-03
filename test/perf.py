import languagemodels
import time
import resource

tests = [
    ("What is the capital of France?", "Paris"),
    ("Lungs are used for respiration in mammals. Do computers have lungs?", "No"),
    ("A game uses a bat and ball. Is it baseball or soccer?", "Baseball"),
    ("What color is the sun, yellow or blue?", "Yellow"),
]

accuracy = 0


def mem_used_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print(f"Memory used before loading models: {mem_used_gb():.1f}GB")

start = time.perf_counter_ns()

languagemodels.chat("Test first run time")

print(f"Initialization time: {(time.perf_counter_ns() - start) / 1e6:.0f}ms")

print(f"Memory used after running chat inference: {mem_used_gb():.1f}GB")

start = time.perf_counter_ns()

for test in tests:
    response = languagemodels.chat(test[0])
    if test[1].lower() in response.lower():
        accuracy += 1 / len(tests)
    print(test[0], response)

print(
    f"Average inference time: {(time.perf_counter_ns() - start)/len(tests)/1e6:.0f}ms"
)
print(f"Overall accuracy: {accuracy}")

print(f"Memory used after running tests: {mem_used_gb():.1f}GB")
