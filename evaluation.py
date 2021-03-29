import json
import torch
import numpy as np
import torch.nn as nn
from math import ceil
from scipy import ndimage
from torch.utils import data

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def pad_image(img, target_size):
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, flip_evaluation, device):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1.0 / 3.0

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_img = torch.from_numpy(padded_img)
            padded_img = padded_img.to(device)
            padded_prediction = net(padded_img)
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def predict_whole(net, image, tile_size, device):
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image.to(device))
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    return prediction


def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, device):
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((H_, W_, classes))
    for scale in scales:
        scale = float(scale)
        #   print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size, device)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:, :, :, ::-1].copy(), tile_size, device)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:, ::-1, :])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
    # print(gt_label.sum())
    # confusion_matrix = np.bincount((class_num * gt_label + pred_label).astype(int), minlength=class_num ** 2).reshape(class_num, class_num)
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def task3_evaluate(model, test_loader, device):
    confusion_matrix = np.zeros((19, 19))

    total = len(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, GT) in enumerate(test_loader):
            print("=========> finished " + str(batch_idx+1) + ", " + str(total-batch_idx-1) + " left")
            #output = predict_multiscale(model, inputs, [1024, 2048], [1.0], 19, False, device)
            output = predict_sliding(model, inputs.numpy(), [1024, 2048], 19, True, device)

            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8).flatten()
            seg_gt = np.asarray(GT.numpy(), dtype=np.int).flatten()

            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, 19)

    fn = confusion_matrix.sum(1) - np.diag(confusion_matrix)
    fp = confusion_matrix.sum(0) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    tn = np.array([confusion_matrix.sum() for i in range(19)]) - confusion_matrix.sum(1) - confusion_matrix.sum(0) + np.diag(confusion_matrix)


    AC_array = (tp + tn) / np.maximum(1.0, fn + fp + tp + tn)
    AC = AC_array.mean()
    SE_array = (tp) / np.maximum(1.0, confusion_matrix.sum(1))
    SE = SE_array.mean()
    SP_array = tn / np.maximum(1.0, tn + fp)
    SP = SP_array.mean()
    DC_array = (2*tp) / np.maximum(1.0, confusion_matrix.sum(0) + confusion_matrix.sum(1))
    DC = DC_array.mean()
    JS_array = (tp) / np.maximum(1.0, confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))

    JS = JS_array.mean()
    

    return AC, SE, SP, DC, JS