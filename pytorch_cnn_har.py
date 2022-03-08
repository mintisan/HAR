# cnn model for HAR with PyTorch
# Reference : https://github.com/harryjdavies/Python1D_CNNs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

from uci_data_loader import *

from pytorch_model1 import *

import netron
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 0 : GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 50, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = ConvNet1D(n_features)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Train the model
    total_step = len(trainX)

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(trainX)
    trainXT = trainXT.transpose(1, 2).float()  # input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
    trainyT = torch.from_numpy(trainy).float()
    testXT = torch.from_numpy(testX)
    testXT = testXT.transpose(1, 2).float()
    testyT = torch.from_numpy(testy).float()

    # fit network
    loss_list = []
    acc_list = []
    acc_list_epoch = []
    loss_list_epoch = []
    for epoch in range(epochs):
        correct_sum = 0
        for i in range(int(np.floor(total_step / batch_size))):  # split data into batches
            trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size]
            trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size]

            # Run the forward pass
            outputs = model(trainXT_seg)
            loss = criterion(outputs, torch.max(trainyT_seg, 1)[1])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = trainyT_seg.size(0)
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(trainyT_seg, 1)
            correct = (predicted == actual).sum().item()
            correct_sum = correct_sum + (correct / total)
            acc_list.append(correct / total)

        acc_list_epoch.append(correct_sum / int(np.floor(total_step / batch_size)))
        loss_list_epoch.append(np.mean(loss_list))

    # evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(testXT)
        _, predictedt = torch.max(test_outputs, 1)
        _, actual = torch.max(testyT, 1)
        total_t = testyT.size(0)
        correct_t = (predictedt == actual).sum().item()
        accuracy = correct_t / total_t

    return accuracy, acc_list_epoch, loss_list_epoch, model


# run an experiment
def run_experiment(data_dir=""):
    # load data set and split into training and testing inputs (X) and outputs (y)
    trainX, trainy, testX, testy = load_dataset(data_dir)

    predict_acc, acc_list_epoch, loss_list_epoch, model = evaluate_model(trainX, trainy, testX, testy)

    # model information
    summary(model, input_size=(32, 9, 128))  # prameter

    d = torch.rand(32, 9, 128)  # model structure
    onnx_path = "pytorch_cnn_har_model.onnx"
    torch.onnx.export(model, d, onnx_path)
    netron.start(onnx_path)

    # summarize history for accuracy
    plt.title("test acc : " + str(np.round(predict_acc, 2)))
    plt.plot(loss_list_epoch, color="blue")
    plt.ylabel('train loss')
    plt.twinx()
    plt.plot(acc_list_epoch, color="orange")
    plt.ylabel('train accuracy')
    plt.xlabel('epoch')
    plt.savefig("pytorch_cnn_har_model_epoch" + '.png')
    # plt.show()
    plt.close()
    plt.clf()


data_dir = r"G:\HAR\HAR"
if __name__ == '__main__':
    # run the experiment
    run_experiment(data_dir)
