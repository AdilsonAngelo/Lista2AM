import knn
from random import randrange


def random_vector(training_set):
    n_records = len(training_set)
    n_features = len(training_set[0])
    return [training_set[randrange(n_records)][i] for i in range(n_features)]


def adjust(w_vector, t_vector, alpha):
    for i in range(len(t_vector) - 1):
        signal = -1 if w_vector[-1] != t_vector[-1] else 1
        error = t_vector[i] - w_vector[i]
        w_vector[i] += signal * alpha * error


def windowed(w, t_vector, closest1, closest2):
    d1 = knn.euclidean(closest1, t_vector)
    d2 = knn.euclidean(closest2, t_vector)
    if d1 == 0 or d2 == 0:
        m = 0
    else:
        m = min(d1/d2, d2/d1)
    s = (1-w)/(1+w)
    return m > s


def lvq1(training_set, prototype_num, alpha=1):
    classes = set()
    for t in training_set:
        classes.add(t[-1])

    if prototype_num < len(classes):
        raise ValueError('Number of prototypes ({}) must be equal or greater then number of classes ({})'.format(
            prototype_num, len(classes)))

    prototypes = [random_vector(training_set) for i in range(prototype_num)]

    for i in range(len(prototypes)):
        prototypes[i][-1] = list(classes)[i % len(classes)]

    for row in training_set:
        closest = knn.get_neighbors(row, prototypes, 1)[0]
        adjust(closest, row, alpha)
    return prototypes


def lvq2(prototypes, training_set, alpha=1, w=.6):
    for row in training_set:

        closest1, closest2 = knn.get_neighbors(row, prototypes, 2)
        if closest1[-1] == closest2[-1]:
            continue
        if closest1[-1] != row[-1] and closest2[-1] != row[-1]:
            continue
        if not windowed(w, row, closest1, closest2):
            continue

        adjust(closest1, row, alpha)
        adjust(closest2, row, alpha)
    return prototypes


def lvq3(prototypes, training_set, alpha=1, epsilon=.6):
    for row in training_set:

        closest1, closest2 = knn.get_neighbors(row, prototypes, 2)
        stabilizer = alpha
        if closest1[-1] == row[-1] and closest2[-1] == row[-1]:
            stabilizer = alpha*epsilon

        adjust(closest1, row, stabilizer)
        adjust(closest2, row, stabilizer)
    return prototypes
