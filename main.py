import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier, EEGNet, custom_loss
from src.utils import set_seed


# @hydra.main(version_base=None, config_path="configs", config_name="config")
# def run(args: DictConfig):
#     set_seed(args.seed)
#     logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
#     if args.use_wandb:
#         wandb.init(mode="online", dir=logdir, project="MEG-classification")

#     # ------------------
#     #    Dataloader
#     # ------------------
#     loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

#     train_set = ThingsMEGDataset("train", args.data_dir)
#     # ThingsMEGDatasetの__itemget__()関数が呼び出される
#     train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
#     val_set = ThingsMEGDataset("val", args.data_dir)
#     val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
#     test_set = ThingsMEGDataset("test", args.data_dir)
#     test_loader = torch.utils.data.DataLoader(
#         test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
#     )

#     # ------------------
#     #       Model
#     # ------------------
#     model = BasicConvClassifier(
#         train_set.num_classes, train_set.seq_len, train_set.num_channels # num_chnnels は入力次元となる
#     ).to(args.device)

#     # ------------------
#     #     Optimizer
#     # ------------------
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) # L2正則化を追加, weight_decay=0.00001

#     # ------------------
#     #   Start training
#     # ------------------  
#     max_val_acc = 0
#     early_stopping_cnt = 0
#     accuracy = Accuracy(
#         task="multiclass", num_classes=train_set.num_classes, top_k=10
#     ).to(args.device)

#     for epoch in range(args.epochs):
#         print(f"Epoch {epoch+1}/{args.epochs}")

#         train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
#         model.train()
#         for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
#             X, y = X.to(args.device), y.to(args.device)

#             y_pred = model(X)
            
#             loss = F.cross_entropy(y_pred, y)
#             train_loss.append(loss.item())
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             acc = accuracy(y_pred, y)
#             train_acc.append(acc.item())

#         model.eval()
#         for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
#             X, y = X.to(args.device), y.to(args.device)
            
#             with torch.no_grad():
#                 y_pred = model(X)
            
#             val_loss.append(F.cross_entropy(y_pred, y).item())
#             val_acc.append(accuracy(y_pred, y).item())

#         print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
#         torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
#         if args.use_wandb:
#             wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
#         if np.mean(val_acc) > max_val_acc:
#             cprint("New best.", "cyan")
#             torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
#             max_val_acc = np.mean(val_acc)
#             early_stopping_cnt = 0
#         else: # Early_stopping
#             early_stopping_cnt += 1

#         if early_stopping_cnt >= args.patience:
#             cprint(f"Validation accuracy did not improve for {args.patience} epochs. Stopping training.", "yellow")
#             break
            
    
#     # ----------------------------------
#     #  Start evaluation with best model
#     # ----------------------------------
#     model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

#     preds = [] 
#     model.eval()
#     for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
#         preds.append(model(X.to(args.device)).detach().cpu())
        
#     preds = torch.cat(preds, dim=0).numpy()
#     np.save(os.path.join(logdir, "submission"), preds)
#     cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

# 

import clip
from src.pre_process import pre_process
import tensorflow as tf
import keras.backend as K
from src.fine_tuning import fine_tuning

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    data_dir = ".\\data"
    dir_path = ".\\data\\images\\Images"
    y_data = 0
    do_pre_process = 2

    if do_pre_process == 2: # 前処理を行う場合データセットを使用する場合
        x_train = torch.load(os.path.join(data_dir, f"train_X.pt"))
        x_train = pre_process(x_train, 'train')
        torch.save(x_train, 'x_train_pre.pt')
        y_train = torch.load(os.path.join(data_dir, f"train_y.pt"))
        y_train = y_data[y_train]
        x_val = torch.load(os.path.join(data_dir, f"val_X.pt"))
        x_val = pre_process(x_val, 'validation')
        torch.save(x_val, 'x_val_pre.pt')
        y_val = torch.load(os.path.join(data_dir, f"val_y.pt"))
        y_val = y_data[y_val]
        x_test = torch.load(os.path.join(data_dir, f"test_X.pt"))
        x_test = pre_process(x_test, 'test')
        torch.save(x_test, 'x_test_pre.pt')
    elif do_pre_process == 1: # 前処理済みデータセットを使用する場合
        x_train = torch.load(f"./x_train_pre.pt").cpu()
        # y_train = torch.load(os.path.join(data_dir, f"train_y.pt"))
        y_train = torch.load(f"x_train_image_features.pt").cpu()
        x_val = torch.load(f"./x_val_pre.pt").cpu()
        # y_val = torch.load(os.path.join(data_dir, f"val_y.pt"))
        y_val = torch.load(f"x_val_image_features.pt").cpu()
        x_test = torch.load(f"./x_test_pre.pt").cpu()
    else: # 前処理済みデータセットを使用しない場合
        x_train = torch.load(f"./data/train_X.pt").cpu()
        y_train = torch.load(os.path.join(data_dir, f"train_y.pt"))
        y_train = y_data[y_train].cpu()
        x_val = torch.load(f"./data/val_X.pt").cpu()
        y_val = torch.load(os.path.join(data_dir, f"val_y.pt"))
        y_val = y_data[y_val].cpu()
        x_test = torch.load(f"./data/test_X.pt").cpu()

    x_train = x_train.numpy()
    x_train = tf.convert_to_tensor(x_train)
    y_train = y_train.numpy()
    y_train = tf.convert_to_tensor(y_train)
    x_val = x_val.numpy()
    x_val = tf.convert_to_tensor(x_val)
    y_val = y_val.numpy()
    y_val = tf.convert_to_tensor(y_val)
    x_test = x_test.numpy()
    x_test = tf.convert_to_tensor(x_test)  

    # ------------------
    #   Start training
    # ------------------  

    # モデルの作成
    model = EEGNet(output=512, Channels=271, Time_seq=281)  # EEGNetモデルの作成
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)

    # モデルのコンパイル
    model.compile(optimizer=optimizer, 
                loss=custom_loss, 
                metrics=['mse'])
    
    # 一つ前のモデルの重みをロード
    # model.load_weights('model.weights.h5')
    
    # モデルの訓練（検証データを指定）
    cprint("Training Start", "cyan")
    model.fit(x_train, y_train, 
            epochs=8,  # エポック数を指定します
            batch_size=128,  # バッチサイズを指定します
            validation_data=(x_val, y_val))  # 検証データを指定します
    cprint("Training Finished", "cyan")

    model.save_weights('model.weights.h5')
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    texts_train = open("./class_y.txt", "r").read().split('\n')

    # モデルの予測
    predictions_tf = model.predict(x_test)
    predictions_torch = torch.from_numpy(predictions_tf).to(device)

    text = clip.tokenize(texts_train).to(device)

    with torch.no_grad():
        # バッチで画像特徴量を取得
        image_features = predictions_torch.unsqueeze(1)  # (N, 1, D) の形状にする
        text_features = model_clip.encode_text(text)  # (M, D) の形状になる

        # ノルムで正規化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 類似度の計算
        # (N, 1, D) @ (D, M) の形状で計算
        result_concatenate = np.empty((0, 1854))
        for i in tqdm(range(image_features.shape[0])):
            probs = (100.0 * image_features[i].to(torch.float) @ text_features.to(torch.float).T).softmax(dim=-1).cpu().numpy()
            result_concatenate = np.concatenate((result_concatenate, probs), axis=0)
        preds = result_concatenate  # 最初の時はそのまま代入

    np.save(os.path.join(logdir, "submission"), preds)#preds.numpy())
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")



if __name__ == "__main__":
    run()
