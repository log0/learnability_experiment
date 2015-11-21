"""
Generate dataset for testing a machine learning algorithm to learn a
simple game of filling in digits.
"""
import csv
import random
import sys

def generate_target(n):
    answer = list(range(1, n + 1))
    random.shuffle(answer)
    return answer

def generate_features(answer):
    return [random.randint(0, 1) and i or 0 for i in answer]

def generate_pairs(n_examples, answer_len):
    targets = [generate_target(answer_len) for i in range(n_examples)]
    pairs = [(generate_features(target), target) for target in targets]
    return pairs

if __name__ == '__main__':
    answer_len = 5
    n_examples = int(sys.argv[1])
    pairs = generate_pairs(n_examples, answer_len)
    
    combined_pairs = []
    for features, target in pairs:
        combined = []
        combined.extend(features)
        combined.extend(target)
        combined_pairs.append(combined)
    combined_pairs = sorted(combined_pairs)
        
    filename = 'train_%d.csv' % (answer_len)
    file_output = open(filename, 'w')
    writer = csv.writer(file_output, lineterminator = '\n')
    writer.writerow([answer_len])
    for combined in combined_pairs:
        writer.writerow(combined)