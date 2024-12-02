import os

EXPERIMENTS = "experiments"

directories = [d for d in os.listdir(
    EXPERIMENTS) if os.path.isdir(os.path.join(EXPERIMENTS, d))]

results = []

for dir in directories:
    log_file_path = os.path.join(EXPERIMENTS, dir, "log.txt")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            lines_with_f1 = [
                line for line in lines if "weighted avg" in line]
            if len(lines_with_f1) > 0:
                f1 = ' '.join(lines_with_f1[0].strip().split()).split(' ')[4]
                results.append((dir, f1))

results.sort(key=lambda x: x[1], reverse=True)
for result in results:
    print(f'Experiment {result[0]} has F1 score: {result[1]}')
