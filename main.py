import csv
import math
import knn
import lvq


def k_fold_dataset(dataset, k):
    n = math.ceil(len(dataset)/k)
    return [dataset[i:i + n] for i in range(0, len(dataset), n)]


csvs = ['cm1', 'kc2']  # , 'jm1']

# PARAMETERS
ks = [1, 3]
proto_nums = [3, 5, 7, 9, 11, 15, 19, 21]


for c in csvs:
    for k in ks:
        csv_file = open(c + '.csv')
        rows = csv.reader(csv_file)
        next(rows)
        dataset = list(rows)

        classes = set()

        for d in dataset:
            for j in range(len(d)-1):
                d[j] = float(d[j])
            d[-1] = True if d[-1] == 'true' else False

        folded_dataset = k_fold_dataset(dataset, 10)

        for proto_num in proto_nums:
            avg_acc_lvq1 = 0
            avg_acc_lvq2 = 0
            avg_acc_lvq3 = 0
            avg_acc_knn = 0
            for i in range(len(folded_dataset)):
                test_set = folded_dataset[i]
                training_set = []
                for j in range(len(folded_dataset)):
                    if j != i:
                        training_set += folded_dataset[j]

                lvq1_proto = lvq.lvq1(training_set, proto_num, .01)
                lvq2_proto = lvq.lvq2(lvq1_proto, training_set, .01)
                lvq3_proto = lvq.lvq3(lvq1_proto, training_set, .01)

                avg_acc_lvq1 += knn.train(lvq1_proto, test_set, k)
                avg_acc_lvq2 += knn.train(lvq2_proto, test_set, k)
                avg_acc_lvq3 += knn.train(lvq3_proto, test_set, k)
                avg_acc_knn += knn.train(training_set, test_set, k)

            avg_acc_lvq1 = 100*avg_acc_lvq1/len(folded_dataset)
            avg_acc_lvq2 = 100*avg_acc_lvq2/len(folded_dataset)
            avg_acc_lvq3 = 100*avg_acc_lvq3/len(folded_dataset)
            avg_acc_knn = 100*avg_acc_knn/len(folded_dataset)

            print("""
DATASET {} - prototype number: {}
lvq1: {:2.2f}%
lvq2: {:2.2f}%
lvq3: {:2.2f}%
{}-nn: {:2.2f}%""".format(c, proto_num, avg_acc_lvq1, avg_acc_lvq2, avg_acc_lvq3, k, avg_acc_knn))
