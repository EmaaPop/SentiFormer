import os
import torch
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.utils import AverageMeter
from models.met import build_model
from core.metric import MetricsTop
from core.new_dataset import Meta
from torch.utils.data import Dataset, DataLoader, Subset


opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))


train_mae, val_mae = [], []


def main():
    opt = parse_opts()

    model = build_model(opt).to(device)
    model.load_state_dict(torch.load('SentiFormer/checkpoint/best/797.pth')['state_dict'])

    dataset = Meta(opt)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False)


    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = MetricsTop().getMetics(opt.datasetName)


    test(model, val_loader, loss_fn, metrics)


def test(model, test_loader, loss_fn, metrics):
    test_pbar = tqdm(test_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data,label in test_pbar:
            img, heu_prompt, text = data[:,2].to(device), data[:,1].to(device), data[:,0].to(device)
            label = label.to(device)
            batchsize = img.shape[0]

            output = model(img, heu_prompt, text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                                   'loss': '{:.5f}'.format(losses.value_avg),
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)




if __name__ == '__main__':
    main()
