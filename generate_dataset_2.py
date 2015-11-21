"""
Generate dataset for testing a machine learning algorithm to learn a
simple game of filling in digits.

This has a minor variation in that the filled in value is always sorted.
Note the pre-populated value remains unchanged.
"""
import csv
import random
import sys

ANSWER_LEN = 6

def generate_target():
    answer = list(range(1, ANSWER_LEN + 1))
    random.shuffle(answer)
    return answer

def generate_features(answer):
    return [random.randint(0, 1) and i or 0 for i in answer]

def generate_unique_answer_from_features(answer):
    features = list(answer)
    remaining_values = list(reversed(sorted(set(range(1, ANSWER_LEN + 1)) - set(features))))
    for i in range(ANSWER_LEN):
        if features[i] == 0:
            features[i] = remaining_values.pop()
    assert len(remaining_values) == 0
    return features

def generate_pairs(n_examples):
    targets = [generate_target() for i in range(n_examples)]
    answers = [generate_features(target) for target in targets]
    pairs = [(answer, generate_unique_answer_from_features(answer)) for answer in answers]
    return pairs

if __name__ == '__main__':
    n_examples = int(sys.argv[1])
    pairs = generate_pairs(n_examples)
    
    combined_pairs = []
    for features, target in pairs:
        combined = []
        combined.extend(features)
        combined.extend(target)
        combined_pairs.append(combined)
    combined_pairs = sorted(combined_pairs)
        
    filename = 'unique_train_%d.csv' % (ANSWER_LEN)
    file_output = open(filename, 'w')
    writer = csv.writer(file_output, lineterminator = '\n')
    writer.writerow([ANSWER_LEN])
    for combined in combined_pairs:
        writer.writerow(combined)