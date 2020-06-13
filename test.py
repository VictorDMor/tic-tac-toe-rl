import numpy as np
number_stats = {}

for i in range(100):
    num = np.random.choice(100)
    if num in number_stats:
        number_stats[num] += 1
    else:
        number_stats[num] = 0
number_stats = {k: v for k, v in sorted(number_stats.items(), key=lambda item: item[1], reverse=True)}
print(number_stats)