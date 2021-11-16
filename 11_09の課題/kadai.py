from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP5(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP5, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = torch.sigmoid(self.l1(x))
        h2 = torch.sigmoid(self.l2(h1))
        h3 = torch.sigmoid(self.l3(h2))
        return torch.sigmoid(self.l4(h3))


if __name__ == '__main__':
    with open('train.dat', 'r') as train:
        train_l = []
        train_d = []
        for r in train:
            train_l.append(r[0])
            train_d.append(r[1:])

    for i in range(len(train_d)):
        train_d[i] = train_d[i].split()
        for j in range(len(train_d[i])):
            train_d[i][j] = train_d[i][j].strip('[,]')
            train_d[i][j] = int(train_d[i][j])
    for i in range(len(train_l)):
        train_l[i] = int(train_l[i])

    with open('test.dat', 'r') as test:
        test_d = []
        for r in test:
            test_d.append(r)

    for i in range(len(test_d)):
        test_d[i] = test_d[i].split()
        for j in range(len(test_d[i])):
            test_d[i][j] = test_d[i][j].strip('[,]')
            test_d[i][j] = int(test_d[i][j])

    """
    sc = StandardScaler()
    sc.fit(train_d)
    train_d_std = sc.transform(train_d)
    test_d_std = sc.transform(test_d)
    """

    model = SVC(kernel='linear', random_state=None)
    model.fit(train_d, train_l)
    pred_train = model.predict(train_d)
    accuracy_train = accuracy_score(train_l, pred_train)
    print('SVM；トレーニングデータに対する正解率： %.2f' % accuracy_train)

    train_predict = model.predict(train_d)
    n = 0
    p = 0
    for i in range(len(train_predict)):
        if train_predict[i] == 1:
            n += 1
            if train_l[i] == 1:
                p += 1
    precision = p / n
    for i in range(len(train_l)):
        if train_l[i] == 1:
            n += 1
            if train_predict[i] == 1:
                p += 1
    recall = p / n

    F_measure = 2 * recall * precision / (recall + precision)
    print('F値_SVM: %.2f' % F_measure)

    predict = model.predict(test_d)
    answerfile = 'test_SVM_ans_1191201079.dat'
    f = open(answerfile, 'w')
    f.close()
    with open(answerfile, 'a') as f:
        for i in range(len(predict)):
            print(predict[i], end=' ', file=f)
            print(test_d[i], end='\n', file=f)


    #MLP
    x = torch.tensor(train_d, dtype=torch.float)
    y = torch.tensor(train_l, dtype=torch.float)
    y = y.unsqueeze(1)

    model = MLP5(33, 3, 1)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    history = []
    max_epoch = 10000
    for epoch in range(max_epoch):
        pred = model(x)
        error = criterion(pred, y)
        history.append(error.item())
        model.zero_grad()
        error.backward()
        optimizer.step()

    """
    history = np.array(history, dtype=np.float32)
    epochs = np.arange(1, max_epoch + 1)
    plt.plot(epochs, history)
    plt.show()
    """
    x_t = torch.tensor(test_d, dtype=torch.float)
    prediction = model(x_t)
    ans = []
    for i in range(prediction.numel()):
        if prediction[i].item() >= 0.5:
            ans.append(1)
        if prediction[i].item() < 0.5:
            ans.append(0)
    # print(ans)
    answerfile = 'test_NNW_ans_1191201079.dat'
    f = open(answerfile, 'w')
    f.close()
    with open(answerfile, 'a') as f:
        for i in range(len(predict)):
            print(ans[i], end=' ', file=f)
            print(test_d[i], end='\n', file=f)

    x_t_N = torch.tensor(train_d, dtype=torch.float)
    prediction_t = model(x_t_N)
    ans = []
    for i in range(prediction_t.numel()):
        if prediction_t[i].item() >= 0.5:
            ans.append(1)
        if prediction_t[i].item() < 0.5:
            ans.append(0)
    n = 0
    p = 0
    for i in range(len(ans)):
        if ans[i] == 1:
            n += 1
            if train_l[i] == 1:
                p += 1
    precision = p / n
    for i in range(len(train_l)):
        if train_l[i] == 1:
            n += 1
            if ans[i] == 1:
                p += 1
    recall = p / n

    F_measure = 2 * recall * precision / (recall + precision)
    print('F値_NNW: %.2f' % F_measure)