import argparse
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from data_loader.msrs_data import MSRS_data
from models.cls_model import Illumination_classifier
from models.common import gradient, clamp
from models.fusion_model import HAIAFusion
from piq import vif_p
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def init_seeds(seed=3407):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def loss_Qabf(fused_image, vis_y_image, inf_image):
    loss_qabf = torch.abs(fused_image - vis_y_image) - torch.abs(fused_image - inf_image)
    return torch.mean(loss_qabf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch HAIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/msrs_train',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='pretrained')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int, metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[3, 7, 50, 1, 3]', type=str, metavar='N', help='loss weight')
    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth', help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')
    parser.add_argument('--cuda', default=True, type=bool, help='use GPU or not')

    args = parser.parse_args()

    init_seeds(args.seed)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'fusion_model':
        model = HAIAFusion().cuda()

        cls_model = Illumination_classifier(input_channels=3)
        cls_model.load_state_dict(torch.load(args.cls_pretrained))
        cls_model = cls_model.cuda()
        cls_model.eval()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.start_epoch, args.epochs):
            lr = args.lr if epoch < args.epochs // 2 else args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            for vis_image, vis_y_image, _, _, inf_image, _ in train_tqdm:
                vis_y_image, vis_image, inf_image = vis_y_image.cuda(), vis_image.cuda(), inf_image.cuda()
                optimizer.zero_grad()

                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)

                pred = cls_model(vis_image)
                day_p, night_p = pred[:, 0], pred[:, 1]
                vis_weight = day_p / (day_p + night_p)
                inf_weight = 1 - vis_weight

                loss_illum = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                                       inf_weight[:, None, None, None] * inf_image) + F.l1_loss(
                    vis_weight[:, None, None, None] * fused_image,
                    vis_weight[:, None, None, None] * vis_y_image)

                loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))

                gradient_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_y_image)))

                loss_vif = 1 - vif_p(fused_image, vis_y_image, data_range=1.0) - vif_p(fused_image, inf_image, data_range=1.0)

                t1, t2, t3, t4, t5 = eval(args.loss_weight)
                loss_qabf_val = loss_Qabf(fused_image, vis_y_image, inf_image)
                total_loss = t1 * loss_illum + t2 * loss_aux + t3 * gradient_loss + t4 * loss_vif + t5 * loss_qabf_val

                train_tqdm.set_postfix(epoch=epoch, loss_illum=t1 * loss_illum.item(), loss_aux=t2 * loss_aux.item(),
                                       gradient_loss=t3 * gradient_loss.item(), loss_vif=t4 * loss_vif.item(),
                                       loss_qabf=t5 * loss_qabf_val.item(), loss_total=total_loss.item())

                total_loss.backward()
                optimizer.step()

            torch.save(model.state_dict(), f'{args.save_path}/fusion_model_epoch_{epoch}.pth')
