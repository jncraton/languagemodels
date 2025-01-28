import languagemodels as lm
import time
import json
import os
import psutil


def mem_used_gb():
    process = psutil.Process(os.getpid())
    bytes = process.memory_info().rss
    gigabytes = bytes * 1e-9
    return gigabytes


print(f"Memory used before loading models: {mem_used_gb():.2f}GB")


print("\n# Completion Test\n")

print(f'{lm.complete("They ran until")=}')

print("\n# Instruction Tests\n")

tests = [
    ("What is the capital of France?", "Paris"),
    ("A game uses a bat and ball. Is it baseball or soccer?", "Baseball"),
    ("Is grass green or blue?", "Green"),
    ("Does a car have more wheels than a bike?", "Yes"),
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

max_ram = lm.config["max_ram"]
print(
    f"Memory used after all tests: {mem_used_gb():.2f}GB (must be under {max_ram:.2f}GB)"
)

# Confirm that we used the right model size and roughly fit in memory constraints
# Note that memory usage will vary between operating systems and specific usage
assert mem_used_gb() < max_ram * 1.10
