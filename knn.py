import math


def euclidean(a, b):
    distance = 0.0
    for i in range(len(a)-1):
        distance += (a[i] - b[i])**2
    return math.sqrt(distance)


def get_neighbors(instance, training_set, k):
    distances = []
    for i in training_set:
        dist = euclidean(i, instance)
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def get_response(neighbors):
    classes = [n[-1] for n in neighbors]
    t = sum(classes)
    f = len(classes)-t
    if f >= t:
        return False
    else:
        return True


def get_accuracy(results):
    correct = 0
    if not results:
        return 0
    for r in results:
        if r[0][-1] == r[1]:
            correct += 1
    res = correct/len(results)
    return res


def train(training_set, test_set, k):
    if k > len(training_set):
        raise ValueError('k value ({}) is too high, dataset only contains {} entries'.format(k, len(training_set)))
        return None
    results = []
    for t in test_set:
        results.append(
            (t, get_response(
                get_neighbors(t, training_set, k))))
    return get_accuracy(results)
