import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import Dataset_Game
from torchvision.utils import save_image
from torchvision.models import resnet50, ResNet50_Weights
import random
import torch.optim as optim
from torch import stack
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_absolute_error
from collections import defaultdict

class Cla_model(nn.Module):
    def __init__(self, args):
        super(Cla_model, self).__init__()
        self.args = args        
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_ftr = self.resnet.fc.in_features
        out_ftr = 9
        self.resnet.fc = nn.Linear(in_ftr, out_ftr, bias=True)
        self.optim      = optim.Adam(self.resnet.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4, 8, 15], gamma=0.1)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.batch_size = args.batch_size
        self.writer = SummaryWriter(args.save_root)
        self.device = args.device
        
    def training_stage(self):
        for i in range(self.args.num_epoch):
            self.resnet.train()
            train_loader = self.train_dataloader()
            total_loss = 0
            pbar = tqdm(train_loader, ncols=120)
            for img, label,_ in pbar:
                img = img.to(self.device)
                label = label.to(self.device)
                # training one step
                self.optim.zero_grad()
                result = self.resnet(img)
                loss = self.ce_criterion(result, label)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                
                self.tqdm_bar('train ', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            print('training loss:', total_loss/len(train_loader))
            self.writer.add_scalar('train/loss', total_loss/len(train_loader), self.current_epoch)
            if (self.current_epoch+1) % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch_{self.current_epoch}.ckpt"))
                
            eval_acc, eval_f1, eval_mae = self.eval()
            self.writer.add_scalar('val/accuracy', eval_acc, self.current_epoch)
            self.writer.add_scalar('val/f1_score', eval_f1, self.current_epoch)
            self.writer.add_scalar('val/mae', eval_mae, self.current_epoch)
            self.current_epoch += 1
            self.scheduler.step()
    
    def store_parameters(self):
        # save args
        with open(os.path.join(self.args.save_root, 'args.yaml'), 'w') as f:
            for k, v in vars(self.args).items():
                f.write(f"{k}: {v}\n")

            
            
    @torch.no_grad()
    def eval(self):
        self.resnet.eval()
        val_loader = self.val_dataloader()
        total_loss = 0
        pbar = tqdm(val_loader, ncols=120)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for (img, label,_) in pbar:
                img = img.to(self.device)
                label = label.to(self.device)
                result = self.resnet(img)
                loss = self.ce_criterion(result, label)
                total_loss += loss.item()
                preds = result.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        f1 = f1_score(all_labels, all_preds, average='weighted')
        mae = mean_absolute_error(all_labels, all_preds)
        print(f"Epoch {self.current_epoch}, val_loss: {total_loss/len(val_loader)},val_f1: {f1},val_mae: {mae}")
        return total_loss/len(val_loader), f1, mae
    
    def train_dataloader(self):
        dataset = Dataset_Game(root=self.args.DR, mode='train')
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True)  
        return train_loader
    
    def val_dataloader(self):
        dataset = Dataset_Game(root=self.args.DR, mode='val')  
        val_loader = DataLoader(dataset,
                                  batch_size=16,
                                  num_workers=self.args.num_workers,
                                  shuffle=False)  
        return val_loader
    
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            # self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.current_epoch = checkpoint['last_epoch']
            
    @torch.no_grad()        
    def test(self):
        self.resnet.eval()
        test_dataset = Dataset_Game(root=self.args.DR, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=self.args.num_workers, shuffle=False)       
        all_labels = []
        all_preds = []
        all_idxs = []
        with torch.no_grad():
            for (img, label,idx) in tqdm(test_loader, ncols=100):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                result = self.resnet(img)
                preds = result.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_idxs.append(list(idx)[0])
        merged_labels, merged_preds = self.merge_labels_preds(all_labels, all_preds, all_idxs)
        f1 = f1_score(merged_labels, merged_preds, average='weighted')
        mae = mean_absolute_error(merged_labels, merged_preds)
        print(f"test_f1: {f1},test_mae: {mae}")
        
        result_path = os.path.join(self.args.save_root,'result.txt' )
        with open(result_path, 'w') as file:
            file.write("appid, pred, label\n")
            for i in range(len(all_idxs)):
                file.write(f"{all_idxs[i]}, {all_preds[i]}, {all_labels[i]}\n")
        
    
    def merge_labels_preds(self,all_labels, all_preds, all_idxs):
        idx_dict = defaultdict(list)
        for idx, label, pred in zip(all_idxs, all_labels, all_preds):
            idx_dict[idx].append((label, pred))
            
        merged_labels = []
        merged_preds = []
        
        for idx, items in idx_dict.items():
            labels, preds = zip(*items)
            avg_label = round(sum(labels) / len(labels))
            avg_pred = round(sum(preds) / len(preds))
            merged_labels.append(avg_label)
            merged_preds.append(avg_pred)

        return merged_labels, merged_preds
        




def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = Cla_model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.test()
    else:
        model.training_stage()
        model.test()
