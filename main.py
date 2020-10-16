# from conf import *
import config
import argparse
from data_loader import NashDataset
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import os
import pandas as pd
import logging
from tqdm import tqdm
from utils import set_logger
import torch.nn as nn
from evaluation import eval
from data_loader import fetch_dataloaders_nash, fetch_dataloaders_debug, \
    fetch_dataloaders_nash_all_features, fetch_dataloaders_alpha_all_features, fetch_dataloaders_ALZ_all_features

def focal_loss(bce_loss, targets, gamma, alpha):
    """Binary focal loss, mean.

    Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
    improvements for alpha.
    :param bce_loss: Binary Cross Entropy loss, a torch tensor.
    :param targets: a torch tensor containing the ground truth, 0s and 1s.
    :param gamma: focal loss power parameter, a float scalar.
    :param alpha: weight of the class indicated by 1, a float scalar.
    """
    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()

def train_single_epoch(model, dataloader, optimizer,criterion,device, focal_loss_flag):
    loss = 0.
    # model.train()
    for i, (train_X, train_y) in enumerate(dataloader):
        train_X = train_X.to(device)
        train_y = train_y.to(device)
        # if i ==0:
        #     print('The batch is on:', train_X.device)
        #     print('The batch is on:', train_y.device)

        train_y = train_y
        optimizer.zero_grad()
        # print(train_X.device)
        # print(train_y.device)
        y_pred = model(train_X)
        # print("Here")
        loss_tmp = criterion(y_pred, train_y)
        if focal_loss_flag:
            loss_f = focal_loss(loss_tmp, train_y, gamma=2, alpha=0.25) ## TODO debug focal loss
        else:
            loss_f = loss_tmp
        # loss_f = loss_tmp
        loss_f.backward()
        optimizer.step()
        loss += loss_f.item()
        if i % 500 == 0:
            # break  ## TODO for debugging
            print("MINIBATCH_loss : {:05.3f}".format(loss_f.item()))
            logging.info("MINIBATCH_loss : {:05.3f}".format(loss_f.item()))
    loss /= (i + 1)
    print("BATCH_loss : {:05.3f}".format(loss))
    logging.info("BATCH_loss : {:05.3f}".format(loss))
    return loss

def evaluate_test(model_name, model, dataloaders, device, summary):
    # rmse, loss = 0., 0.
    # with torch.no_grad():
    # model.eval()
    for split in list(dataloaders.keys()):
        y_prob_list = []
        y_tag_list = []
        y_true_list = []
        dataloader = dataloaders[split]
        for i, (train_X, train_y) in enumerate(dataloader):
            test_X = train_X.to(device)
            # test_y = train_y.to(device)
            if model_name == 'stnb.pth':
                y_prob_tmp = torch.zeros(10, test_X.shape[0], 2)
                for i in range(10):
                    y_prob = model(test_X)
                    y_prob_tmp[i,:,:] =  y_prob
                y_prob = y_prob_tmp.mean(0)
            else:
                y_prob = model(test_X)
            y_tag = torch.argmax(y_prob, dim=1)
            y_prob = y_prob[:, 1]
            y_prob_list.extend(y_prob.view(-1).cpu().detach().numpy())
            y_tag_list.extend(y_tag.view(-1).cpu().detach().numpy())
            y_true_list.extend(train_y.view(-1).detach().numpy())
        # eval_Y = dataloaders[split].dataset.Y
        # eval_X = eval_X.to(device)
        #    eval_Y = eval_Y.to(device)
        # y_prob = model(eval_X)
        #    print('Dataset.X is on:', eval_X.device)
        #    print('Dataset.Y is on:', eval_Y.device)
        # y_prob = y_prob.cpu()
        # # y_tag = torch.round(y_prob)
        # y_tag = torch.argmax(y_prob, dim=1)
        # y_prob = y_prob[:, 1]
        # y_true = eval_Y
        accuracy, precision, recall, mcc, f1, kappa, roc_auc, pr_auc, conf_matrix = eval(y_true_list, y_tag_list,
                                                                                         y_prob_list)
        tn, fp, fn, tp = conf_matrix.ravel()
        # summary.append(model_name, split, accuracy, precision, recall, f1, kappa, roc_auc, pr_auc, tn, fp, fn, tp)

        summary["Model"].append(model_name)
        summary["Data split"].append(split)
        summary["Accuracy"].append(accuracy)
        summary["Precision"].append(precision)
        summary["Recall"].append(recall)
        summary["mcc"].append(mcc)

        summary["F1 Score"].append(f1)
        summary["Cohens Kappa"].append(kappa)
        summary["ROC AUC"].append(roc_auc)
        summary["PR AUC"].append(pr_auc)
        summary["True negtive (TN)"].append(tn)
        summary["False positive (FP)"].append(fp)
        summary["False negtive (FN)"].append(fn)
        summary["True positive (TP)"].append(tp)



    return copy.deepcopy(summary)


def evaluate(model, dataloader, split, device):
    # rmse, loss = 0., 0.
    # with torch.no_grad():
    # eval_X = dataloader.dataset.X
    # eval_Y = dataloader.dataset.Y
    # eval_X = eval_X.to(device)
    y_prob_list =[]
    y_tag_list = []
    y_true_list = []
#    eval_Y = eval_Y.to(device)
    for i, (train_X, train_y) in enumerate(dataloader):
        test_X = train_X.to(device)
        # test_y = train_y.to(device)

        y_prob = model(test_X)
        y_tag = torch.argmax(y_prob, dim=1)
        y_prob = y_prob[:, 1]
        y_prob_list.extend(y_prob.view(-1).cpu().detach().numpy())
        y_tag_list.extend(y_tag.view(-1).cpu().detach().numpy())
        y_true_list.extend(train_y.view(-1).detach().numpy())

#    print('Dataset.X is on:', eval_X.device)
#    print('Dataset.Y is on:', eval_Y.device)
#     y_prob = y_prob.cpu()
#     y_tag = torch.argmax(y_prob, dim=1)
#     y_prob = y_prob[:,1]
#     # y_tag = torch.round(y_prob)
#     y_true = eval_Y
    accuracy, precision, recall, mcc, f1, kappa,roc_auc, pr_auc, conf_matrix = eval(y_true_list, y_tag_list, y_prob_list)
    tn, fp, fn, tp = conf_matrix.ravel()
    logging.info(split.upper() + "_accuracy : {:05.3f}".format(accuracy))
    logging.info(split.upper() + "_precision : {:05.3f}".format(precision))
    logging.info(split.upper() + "_recall : {:05.3f}".format(recall))
    logging.info(split.upper() + "_f1 : {:05.3f}".format(f1))
    logging.info(split.upper() + "_kappa : {:05.3f}".format(kappa))
    logging.info(split.upper() + "_roc_auc : {:05.3f}".format(roc_auc))
    logging.info(split.upper() + "_pr_auc : {:05.3f}".format(pr_auc))
    logging.info(split.upper() + "_TN : {}".format(tn))
    logging.info(split.upper() + "_FP : {}".format(fn))
    logging.info(split.upper() + "_FN : {}".format(fn))
    logging.info(split.upper() + "_TP : {}".format(tp))
    ## TODO save to dataframe csv

    return f1

def train_and_evaluate(model, dataloaders, optimizer, n_epochs,criterion, device,focal_loss_flag):
    dl_train = dataloaders['train']
    dl_val = dataloaders['val']

    best_mcc = float(-2)
    best_state = None
    with tqdm(total=n_epochs) as t:
        for i in range(n_epochs):
            if i ==40:
                print('here')
            loss = train_single_epoch(model, dl_train, optimizer,criterion, device, focal_loss_flag)
            mcc_val = evaluate(model, dl_val,'val', device)
            # f1_val=0
            is_best = mcc_val >= best_mcc
            if is_best:
                best_mcc = mcc_val
                best_state = copy.deepcopy(model.state_dict())
                logging.info("Found new best error at {}".format(i))
            t.set_postfix(loss_and_best_mcc='{:05.3f} and {:05.3f}'.format(
                loss, best_mcc))
            print('\n')
            t.update()
    # evaluate(model, dl_val, 'val')
    return best_state

def main(args):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu' if 'cpu' in args.device else device)
    print('Selected device is:', device)
    # device = torch.device('cpu')

    # logger configuration
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    set_logger(os.path.join(log_dir, 'train.log'))


    dic_model_conf = getattr(config, "DIC_{0}_CONF".format(args.model.upper()))
    if args.all:
        if args.dataset == 'nash':
            dic_model_conf['vocab_size'] = 24285 # 30692 # 24285 for nash all
        else:
            dic_model_conf['vocab_size'] = 30692
    model = config.DIC_MODEL[args.model](dic_model_conf, device).to(device)
    if args.debug:
        dataloaders = fetch_dataloaders_debug(['train','test','val'], args.batch_size)
    elif args.all:
        if args.dataset == 'nash':
            dataloaders = fetch_dataloaders_nash_all_features(['train','test','val'], args.batch_size)
        else:
            dataloaders = fetch_dataloaders_ALZ_all_features(['train','test','val'], args.batch_size)
    else:
        dataloaders = fetch_dataloaders_nash(['train','test','val'], args.batch_size)


    my_criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss() #nn.BCELoss()  ## TODO bce loss example nn.CrossEntropyLoss()
    my_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



    best_state = train_and_evaluate(model, dataloaders,
                                    n_epochs=args.num_epoch,
                                    optimizer=my_optimizer,
                                    criterion=my_criterion,
                                    device=device,
                                    focal_loss_flag=args.focal_loss)

    # save a model
    model_name = '{}.pth'.format(args.model)
    state = {
        'dic_model_conf': dic_model_conf,
        'state_dict': best_state,
        'n_epoch': args.num_epoch,
        'lr': args.lr
    }
    torch.save(state, os.path.join(log_dir, model_name))

    total_summary = {
        "Model": [],
        "Data split": [],
        "Accuracy":[],
        "Precision":[],
        "Recall":[],
        "mcc": [],
        "F1 Score":[],
        "Cohens Kappa":[],
        "ROC AUC": [],
        "PR AUC": [],
        "True negtive (TN)":[],
        "False positive (FP)":[],
        "False negtive (FN)":[],
        "True positive (TP)":[]
    }


    ### testing=====================
    summary_path = os.path.join(log_dir, '{}_summary.csv'.format(model_name))
    model.load_state_dict(best_state)

    if args.all:
        if args.dataset == 'nash':
            dataloaders = fetch_dataloaders_nash_all_features(['test', 'val'], 128)
        else:
            dataloaders = fetch_dataloaders_ALZ_all_features(['test', 'val'], 128)
    else:
        dataloaders = fetch_dataloaders_nash(['test', 'val'])
    total_summary = evaluate_test(model_name, model, dataloaders, device, copy.deepcopy(total_summary))
    total_summary = pd.DataFrame(total_summary,columns=total_summary.keys())
    total_summary.to_csv(summary_path,sep='\t',index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # lr CTransformer
    # lstm dipole
    parser.add_argument('--model', type = str, default = 'stn', help = 'model_name')
    parser.add_argument('--bnn', type = bool, default = True, help = 'bayesian flag')
    parser.add_argument('--dataset', type = str, default = 'alz', help = 'chosen dataset')
    parser.add_argument('--log_dir', type = str, default = 'experiments/0908_alz-location', help = 'logging directory')
   # parser.add_argument('--train_data', type = str, default = 'nash_small_train.csv', help = 'training dataset')
#    parser.add_argument('--test_data', type = str, default = 'nash_small_test.csv', help = 'testing dataset')
    parser.add_argument('--num_epoch', type = int, default = 30, help = '# epochs')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'batch size')
    parser.add_argument('--focal_loss', type = bool, default = False, help = 'whether use focal loss')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--debug', type = bool, default = False, help = 'simple scenario debugging')
    parser.add_argument('--all', type = bool, default = True, help = 'All features')
    parser.add_argument('--device', type = str, default = "cuda", help = 'gpu device')
    args = parser.parse_args()
    # num_epoch
    print(args)
    main(args)



