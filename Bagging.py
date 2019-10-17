import csv
import math
import random


class Bagging:
    def __init__(self):
        self.datatrain = []
        self.datatest = []
        self.c1 = 0
        self.c2 = 0
        self.rata_c1_x1 = 1
        self.rata_c1_x2 = 1
        self.rata_c2_x1 = 1
        self.rata_c2_x2 = 1
        self.std_c1_x1 = 1
        self.std_c1_x2 = 1
        self.std_c2_x1 = 1
        self.std_c2_x2 = 1
        self.output = []

    def open_datatrain(self):
        with open('Trainset.csv') as csvfile:
            next(csvfile)
            spamreader = csv.reader(csvfile)
            for row_train in spamreader:
                self.datatrain.append(row_train)
        csvfile.close()

    def open_datatest(self):
        with open('Testset.csv') as csvfile:
            next(csvfile)
            spamreader = csv.reader(csvfile)
            for row_test in spamreader:
                self.datatest.append(row_test)
        csvfile.close()

    def prob_class(self, a):
        jum_c1_x1 = 0
        jum_c1_x2 = 0
        jum_c2_x1 = 0
        jum_c2_x2 = 0
        for data in a:
            if data[2] == "1":
                self.c1 += 1
                jum_c1_x1 += float(data[0])
                jum_c1_x2 += float(data[1])
            else:
                self.c2 += 1
                jum_c2_x1 += float(data[0])
                jum_c2_x2 += float(data[1])
        self.rata_c1_x1 = jum_c1_x1 / self.c1
        self.rata_c1_x2 = jum_c1_x2 / self.c1
        self.rata_c2_x1 = jum_c2_x1 / self.c2
        self.rata_c2_x2 = jum_c2_x2 / self.c2

    def std_class(self, a):
        c1_x1 = 0
        c1_x2 = 0
        c2_x1 = 0
        c2_x2 = 0
        for data in a:
            if data[2] == "1":
                c1_x1 = c1_x1 + ((float(data[0]) - self.rata_c1_x1) ** 2)
                c1_x2 = c1_x2 + ((float(data[1]) - self.rata_c1_x2) ** 2)
            else:
                c2_x1 = c2_x1 + ((float(data[0]) - self.rata_c2_x1) ** 2)
                c2_x2 = c2_x2 + ((float(data[1]) - self.rata_c2_x2) ** 2)
        self.std_c1_x1 = math.sqrt(c1_x1 / (self.c1 - 1))
        self.std_c1_x2 = math.sqrt(c1_x2 / (self.c1 - 1))
        self.std_c2_x1 = math.sqrt(c2_x1 / (self.c2 - 1))
        self.std_c2_x2 = math.sqrt(c2_x2 / (self.c2 - 1))

    def calc_naive_bayes(self, std1, std2, rata1, rata2, x1, x2, c,c1):
        constanta = 0.000001
        n1 = (1/(math.sqrt(2*math.pi))*std1)*math.exp(-((x1-rata1)**2/(2*(std1**2)+ constanta)))
        n2 = (1/(math.sqrt(2*math.pi))*std2)*math.exp(-((x2-rata2)**2/(2*(std2**2)+ constanta)))
        return (c1/len(c))*n1*n2

    def naive_bayes(self, a):
        n = []
        for i in range(len(self.datatest)):
            nb1 = self.calc_naive_bayes(self.std_c1_x1, self.std_c1_x2, self.rata_c1_x1, self.rata_c1_x2, float(self.datatest[i][0]), float(self.datatest[i][1]), a, self.c1)
            nb2 = self.calc_naive_bayes(self.std_c2_x1, self.std_c2_x2, self.rata_c2_x1, self.rata_c2_x2, float(self.datatest[i][0]), float(self.datatest[i][1]), a, self.c2)
            if nb1 > nb2:
                n.append(1)
            else:
                n.append(2)
        return n

    def create_bootstrap(self):
        bootstrap = []
        for i in range(187):
            x = random.choice(self.datatrain)
            bootstrap.append(x)
        return bootstrap

    def count_class(self, a):
        matrix = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
        hasil = []
        for data in matrix:
            kelas_1 = data.count(1)
            kelas_2 = data.count(2)
            if kelas_1 > kelas_2:
                kelas_hasil = "1"
            else:
                kelas_hasil = "2"
            hasil.append(kelas_hasil)
        return hasil

    def create_model(self):
        a = []
        for i in range(5):
            bootstrap = self.create_bootstrap()
            self.prob_class(bootstrap)
            self.std_class(bootstrap)
            a.append(self.naive_bayes(bootstrap))
        self.output.append(self.count_class(a))

    def print_hasil(self):
        with open('Tebakan.csv', 'w') as csvFile:
            text = ""
            for x in self.output[0]:
                text += x + "\n"
            csvFile.write(text)
        csvFile.close()


if __name__ == '__main__':
    e_learning = Bagging()
    e_learning.open_datatrain()
    e_learning.open_datatest()
    e_learning.prob_class(e_learning.datatrain)
    e_learning.std_class(e_learning.datatrain)
    e_learning.naive_bayes(e_learning.datatrain)
    e_learning.create_model()
    e_learning.print_hasil()
