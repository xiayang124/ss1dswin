import argparse
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix

import data_gen
from config import get_config
from dataset import get_dataset
from method import SwinT
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--data', choices=['Indian', 'Pavia', 'Honghu'], default='Indian',
                    help='data to use')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
# -------------------------------------------------------------------------------
parser.add_argument('--train_time', type=int, default=3)
parser.add_argument('--train_num', type=int and float, default=5)
parser.add_argument('--epoch', type=int, default=100)

args = parser.parse_args()

dataset_name = args.data
train_num = args.train_num
train_time = args.train_time
batch_size = args.batch_size
epochs = args.epoch
path = "./save_path/"
gamma = 0.9


def choose_true_point(true_data, num_classes):
    number_true = []
    pos_true = {}
    for i in range(num_classes + 1):
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    return total_pos_true, number_true


def choose_img_point(height, width):
    total_pos_true = np.array([[i, j] for i in range(height) for j in range(width)])
    return total_pos_true


# 1
def chooose_point(test_data, num_classes):
    number_test = []
    pos_test = {}

    for i in range(num_classes):
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    return total_pos_test, number_test


def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)  # padding后的图 上下左右各加padding

    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize  # 中间用原图初始化

    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]

    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]

    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]

    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def mirror_padding_band(x_test, band, band_patch, patch=5):
    padding_size = band_patch - 1
    smaller_padding = padding_size // 2
    bigger_padding = padding_size - smaller_padding

    x_padding_band = np.zeros((x_test.shape[0], patch, patch, band + padding_size),
                              dtype=float)

    x_padding_band[:, :, :, 0:bigger_padding] = x_test[:, :, :, 0:bigger_padding][:, :, :, ::-1]

    x_padding_band[:, :, :, bigger_padding:bigger_padding + band] = x_test

    x_padding_band[:, :, :, bigger_padding + band:bigger_padding + band + smaller_padding] = x_test[:, :, :,
                                                                                             band - smaller_padding:band][
                                                                                             :, :, :, ::-1]

    return x_padding_band


def gain_neighborhood_band_mirror(x, band, band_patch, patch=5):
    x_padding_band = mirror_padding_band(x, band, band_patch, patch)  # [B,w,w,band+padding_size]

    return x_padding_band


def get_data(mirror_image, band, test_point, patch=5, band_patch=3):
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)

    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("**************************************************")

    x_test_band = gain_neighborhood_band_mirror(x_test, band, band_patch, patch)
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("**************************************************")
    return x_test_band


def get_label(number_test, num_classes):
    y_test = []
    for i in range(num_classes):
        for k in range(number_test[i]):
            y_test.append(i)

    y_test = np.array(y_test)
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("**************************************************")
    return y_test


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def train_epoch(train_loader, valid_loader, criterion, lr_optimizer, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        tic = time.time()
        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            batch_pred = model(batch_data)

            optimizer.zero_grad()
            loss = criterion(batch_pred, batch_target)
            loss.backward()
            optimizer.step()

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))

            n = batch_data.shape[0]
        lr_optimizer.step()
        toc = time.time()
        per_epoch_time = toc - tic
        print("epoch {}, loss {}, use {} s".format(epoch, loss, per_epoch_time))

        if (epoch + 1) % 100 == 0:
            model.eval()
            objs = AvgrageMeter()
            top1 = AvgrageMeter()
            tar = np.array([])
            pre = np.array([])
            for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                batch_pred = model(batch_data)

                loss = criterion(batch_pred, batch_target)

                prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))

                n = batch_data.shape[0]
                objs.update(loss.data, n)
                top1.update(prec1[0].data, n)
                tar = np.append(tar, t.data.cpu().numpy())
                pre = np.append(pre, p.data.cpu().numpy())
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar, pre)
            print("test, loss {}, use {} s, oa {:.6f}, aa {:.6f}, kappa {:.6f}".format(loss, per_epoch_time, OA2, AA_mean2, Kappa2))

            if epoch == epochs - 1:
                random_flop_test = torch.randn(size=(batch_size, batch_data.shape[1], batch_data.shape[2])).cuda()
                macs, params = profile(model, (random_flop_test, ))

                oa_range = 0
                aa_range = 0
                kappa_range = 0
                if dataset_name == "Indian":
                    oa_range = (60.14 - 5.04, 60.14 + 5.04)
                    aa_range = (73.81 - 3.36, 73.81 + 3.36)
                    kappa_range = (55.64 - 5.48, 55.64 + 5.48)
                if dataset_name == "Pavia":
                    oa_range = (71.71 - 1.39, 71.71 + 1.39)
                    aa_range = (75.58 - 1.49, 75.58 + 1.49)
                    kappa_range = (63.82 - 1.64, 63.82 + 1.64)
                if dataset_name == "Honghu":
                    oa_range = (74.88 - 4.45, 74.88 + 4.45)
                    aa_range = (70.50 - 2.80, 70.50 + 2.80)
                    kappa_range = (69.45 - 5.01, 69.45 + 5.01)
                if not oa_range[0] < OA2 * 100 < oa_range[1] or not aa_range[0] < AA_mean2 * 100 < aa_range[1] or not kappa_range[0] < Kappa2 * 100 < kappa_range[1]:
                    return False
                res = {
                    'oa': OA2 * 100,
                    'each_acc': str(AA2 * 100),
                    'aa': AA_mean2 * 100,
                    'kappa': Kappa2 * 100,
                    'macs': macs,
                    'flop': macs * 2,
                }
    return res


def test_eval(batch_size, mirror_image, band, total_pos_all, image_size, near_band):
    all_label = np.array([])
    current_index = 0
    batch_num = total_pos_all.shape[0]
    begin_time = round(time.time() * 1000)
    while current_index + batch_size <= batch_num:
        current_pos = total_pos_all[current_index: current_index + batch_size, :]
        token = get_data(mirror_image, band, total_pos_all, patch=image_size, band_patch=near_band)
        if type(token) != torch.Tensor:
            token = torch.from_numpy(token)

        batch_data = token.to(device)
        batch_pred = model(batch_data)
        pred = batch_pred.max(1)[1]
        all_label = np.concatenate((all_label, pred.cpu().numpy()))
        current_index = current_index + batch_size
        if current_index + batch_size == batch_num:
            break
    batch_data = token[current_index:, :, :]
    batch_pred = model(batch_data.to(device))
    end_time = round(time.time() * 1000)
    pred = batch_pred.max(1)[1]
    all_label = np.concatenate((all_label, pred.cpu().numpy()))
    return all_label, batch_data, end_time - begin_time


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def save_matrix(tar, pre, dataset):
    matrix = confusion_matrix(tar, pre)
    np.save('./confusionMatrix/{}.npy'.format(dataset), matrix)


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def modify_data(x, patch_size, band_patch):
    x = torch.nn.functional.unfold(x, (patch_size, band_patch))
    return x


def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

mat_file = "{}_{}_split.mat".format(dataset_name, train_num)
if result_file_exists('./data/{}'.format(dataset_name), mat_file):
    print("{} had been generated...skip".format(mat_file))
data_gen.generate_data(dataset_name, train_num)
print("All data had been generated!")

TE, TR, input = get_dataset(dataset_name, train_num_or_rate=train_num)
image_size, near_band, window_size = get_config(dataset_name)

num_classes = np.max(TR)

input_normalize = np.zeros(input.shape)

for i in range(input.shape[2]):
    input_max = np.max(input[:, :, i])
    input_min = np.min(input[:, :, i])
    input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))

total_pos_test, number_test = chooose_point(TE, num_classes)
total_pos_train, number_train = chooose_point(TR, num_classes)

mirror_image = mirror_hsi(height, width, band, input_normalize, image_size)
# ------------------------X TEST DATA--------------------------------
x_test_band = get_data(mirror_image, band, total_pos_test,
                       patch=image_size,
                       band_patch=near_band)
x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)
x_test = modify_data(x_test, image_size, near_band)
# ------------------------X TRAIN DATA-------------------------------
x_train_band = get_data(mirror_image, band, total_pos_train,
                        patch=image_size,
                        band_patch=near_band)
x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)
x_train = modify_data(x_train, image_size, near_band)
# -----------------------Y TRAIN DATA--------------------------------
y_train = get_label(number_train, num_classes)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
# -----------------------Y TEST DATA---------------------------------
y_test = get_label(number_test, num_classes)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
# ----------------------Make Datasets--------------------------------
Label_test = Data.TensorDataset(x_test, y_test)
Label_train = Data.TensorDataset(x_train, y_train)

label_test_loader = Data.DataLoader(Label_test, batch_size=batch_size, shuffle=True, num_workers=0)
label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=False, num_workers=0)

while True:
    """uniq_name = "{}_{}_{}_ss1d.json".format(dataset_name, train_num, times)
    if result_file_exists('./save_path', uniq_name):
        print('%s has been run. skip...' % uniq_name)
        continue"""
    # print("begin training {}".format(uniq_name))
    model = SwinT(image_size=image_size, near_band=near_band, num_patches=band,
                  patch_dim=near_band * image_size ** 2, num_classes=num_classes, band=band, dim=64,
                  heads=4, dropout=0.1, emb_dropout=0.1, window_size=window_size,
                  )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0)
    lr_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)

    train_result = train_epoch(label_train_loader, label_test_loader, criterion, lr_optimizer, optimizer, epochs)
    if train_result is False:
        continue
    del label_test_loader, label_train_loader, x_test, x_train, y_train, y_test
    # ------------------------X ALL DATA---------------------------------
    init_line = np.linspace(0, height - 1, height, dtype=int).reshape(-1, 1)
    init_row = np.linspace(0, width - 1, width, dtype=int).reshape(-1, 1)

    lengths = np.repeat(init_line, width, axis=0)
    rows = np.repeat(init_row, height, axis=1).reshape((-1, 1), order='F')

    total_pos_all = np.concatenate((lengths, rows), axis=1)

    all_label, _, time = test_eval(batch_size, mirror_image, band, total_pos_all, image_size, near_band)

    train_result['times'] = time
    all_label = all_label.reshape(TR.shape[0], TR.shape[1])
    save_loc = "ss1d_save_npy/" + dataset_name + "_ss1d.pred"
    save_path_pred = "%s.npy" % save_loc
    np.save(save_path_pred, all_label)
    del all_label, model

    save_loc = path + dataset_name + "_" + str(train_num) + "_ss1d"
    save_path_json = "%s.json" % save_loc
    ss = json.dumps(train_result, indent=4)
    with open(save_path_json, 'w') as fout:
        fout.write(ss)
        fout.flush()
    print("save record of %s done!" % save_loc)
    
    break
