import os
import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from core.dataset import Meta
from tensorboardX import SummaryWriter
from model.met import build_model
from core.metric import MetricsTop
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

best = None

beste = None
bestm = None

opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))

train_mae, val_mae = [], []

def main():
    opt = parse_opts()
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))
    
    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)

    save_path = os.path.join(opt.models_save_root,  opt.project_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    model = build_model(opt).to(device)

    # dataLoader = MMDataLoader(opt)
    dataset = Meta(opt)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=0)
    # train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)


    for epoch in range(1, opt.n_epochs+1):
        train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics)
        evaluate(model, val_loader, optimizer, loss_fn, epoch, writer, save_path, metrics)
        # if opt.is_test is not None:
        #     test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics)
        scheduler_warmup.step()
    print('best epoch is:')
    print(beste)
    print('best metric is:')
    print(bestm)
    writer.close()


def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for data,label in train_pbar:
        img, heu_prompt, text = data[:,2].to(device), data[:,1].to(device), data[:,0].to(device)
        label = label.to(device)
        batchsize = img.shape[0]

        output= model(img, heu_prompt, text)
        # maeloss = torch.nn.L1Loss(reduction='mean')
        # output = torch.nn.Softmax(dim=1)(output)
        # a_pred = torch.nn.Softmax(dim=1)(a_logits)
        # v_pred = torch.nn.Softmax(dim=1)(v_logits)
        # t_pred = torch.nn.Softmax(dim=1)(t_logits)
        # a_tcp ,_= torch.max(a_pred*label, dim=1,keepdim=True)
        # v_tcp,_ = torch.max(v_pred*label, dim=1,keepdim=True)
        # t_tcp,_ = torch.max(t_pred*label, dim=1,keepdim=True)
        # print(a_confidence.shape)
        # exit(0)

        # tcp_pred_loss = maeloss(a_confidence,a_tcp.detach())+maeloss(v_confidence,v_tcp.detach())+maeloss(t_confidence,t_tcp.detach())
        loss = loss_fn(output, label)


        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('train: ', train_results)
    train_mae.append(train_results['Accuracy'])

    writer.add_scalar('train/loss', losses.value_avg, epoch)



def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics):
    test_pbar = tqdm(eval_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data,label in test_pbar:
            img, heu_prompt, text = data[:,2].to(device), data[:,1].to(device), data[:,0].to(device)
            label = label.to(device)
            batchsize = img.shape[0]

            output= model(img, heu_prompt, text)
            output = torch.nn.Softmax(dim=1)(output)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)
        global best,beste,bestm
        if best is None or test_results['Accuracy'] > best:
            best = test_results['Accuracy']
            beste = epoch
            bestm = test_results

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)

        save_model(save_path, epoch, model, optimizer)


def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics):
    test_pbar = tqdm(test_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data,label in test_pbar:
            img, heu_prompt, text =data[:,2].to(device), data[:,1].to(device), data[:,0].to(device)
            label = label.to(device)
            batchsize = img.shape[0]

            output = model(img, heu_prompt, text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        writer.add_scalar('test/loss', losses.value_avg, epoch)

if __name__ == '__main__':
    main()