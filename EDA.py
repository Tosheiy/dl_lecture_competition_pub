import torch
from src.datasets import ThingsMEGDataset
import mne
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gc
from PIL import Image
import clip
import os
from src.fine_tuning import fine_tuning

# data_path = mne.datasets.brainstorm.bst_auditory.data_path()
# subject = "bst_auditory"
# subjects_dir = data_path / "subjects"
# raw_fname1 = r"C:\Users\asf72\ansel\Document\ld.ds"
# raw = mne.io.read_raw_ctf(raw_fname1)

# kwargs = dict(eeg=False, coord_frame="meg", show_axes=True, verbose=True)
# raw = mne.io.read_raw_ctf(
#     mne.datasets.spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
# )
# fig = mne.viz.plot_alignment(raw.info, meg=("helmet", "sensors", "ref"), **kwargs)
# mne.viz.set_3d_title(figure=fig, title="CTF 275")
# plt.show()

# # 例としてSPMデータセットを使用
# raw = mne.io.read_raw_ctf(
#     mne.datasets.spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
# )

# # 電極の座標を表示
# for dig in raw.info['chs']:
#     print(dig['ch_name'], dig['loc'])  # 電極名と座標を表示

# for ch in raw.info['chs']:
#     print(ch['ch_name'])

# for dig in raw.info['dig']
# 
from src.pre_process1 import preprocess_meg

orig_sample_rate = 200  # 元のサンプリングレート
new_sample_rate = 120   # 新しいサンプリングレート
tmin = -0.5  # エポックの開始時間（秒）
tmax = 1.0   # エポックの終了時間（秒）
data_dir = ".\\data"

x_train = torch.load(os.path.join(data_dir, f"val_X.pt"))

# データの前処理
preprocessed_data = preprocess_meg(x_train, orig_sample_rate, new_sample_rate, tmin, tmax)

exit()
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

exit()
import tensorflow as tf
def custom_loss(y_true, y_pred):
    t = 0.5
    lamda = 0.8
    # 正規化
    y_true_normalize = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_normalize = tf.nn.l2_normalize(y_pred, axis=1)

    # norm 
    y_true_norm = tf.norm(y_true_normalize, axis=1)
    y_pred_norm = tf.norm(y_pred_normalize, axis=1)

    

    # ペアワイズコサイン類似度
    # size = tf.einsum('i,j->ij', y_pred_norm, y_true_norm)
    # vector = tf.matmul(y_pred_normalize, y_true_normalize, transpose_b=True)
    # cosine_similarity_matrix = tf.divide(vector, size)
    cosine_similarity_matrix = tf.matmul(y_pred_normalize, y_true_normalize, transpose_b=True)

    print(cosine_similarity_matrix)
    
    # expを取ってスコアを計算
    exp_scores = tf.exp(cosine_similarity_matrix / t)

    

    # # 対角要素は除去（対角の部分には相関があるので、損失には不要なケースがある）
    # exp_scores -= tf.linalg.diag(tf.linalg.diag_part(exp_scores))

    # 対角成分を抽出
    score_pred_molecule = tf.linalg.diag_part(exp_scores)
    

    # 対角要素を除いて各行ごとの合計
    score_pred_l = tf.reduce_sum(exp_scores, axis=1)
    
    score_pred_r = tf.reduce_sum(exp_scores, axis=0)
    
    score_pred_r = tf.transpose(score_pred_r)
    
    frac_l = tf.divide(score_pred_molecule, score_pred_l)
    frac_r = tf.divide(score_pred_molecule, score_pred_r)

    
    
    # 損失を計算
    LossClip = -1 * tf.reduce_mean(tf.math.log(frac_l) + tf.math.log(frac_r))
     
    return LossClip

#     print(dig['kind'], dig['r'])
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
# texts_train = open("./class_y.txt", "r").hjhjh().split('\n')

file_path_1 = './data/images/Images/aardvark/aardvark_01b.jpg'
file_path_2 = './data/images/Images/aardvark/aardvark_02s.jpg'

image_1 = preprocess(Image.open(file_path_1)).unsqueeze(0).to(device)
image_2 = preprocess(Image.open(file_path_2)).unsqueeze(0).to(device)

image_features1 = model_clip.encode_image(image_1)
image_features1 = image_features1.cpu().detach().numpy()
image_features1 = tf.convert_to_tensor(image_features1)
image_features1 = tf.reshape(image_features1, (1, 512))
image_features2 = model_clip.encode_image(image_2)
image_features2 = image_features2.cpu().detach().numpy()
image_features2 = tf.convert_to_tensor(image_features2)
image_features2 = tf.reshape(image_features2, (1, 512))



print(custom_loss(image_features1, image_features2))
exit()


y_ = torch.load("./tensor.pt")
y = torch.load("./data/val_y.pt")
class_y = open("./class_y.txt", "r").read().split('\n')
text = clip.tokenize(class_y).to(device)
idx = [83, 232, 121, 12, 1]
predictions_tf = y_[y]

predictions_torch = predictions_tf.to(device)

# for i in range(predictions_torch.shape[0]):
#     with torch.no_grad():
#         image_features = predictions_torch[i].unsqueeze(0)
#         text_features = model_clip.encode_text(text)

#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         # 類似度の計算
#         probs = (100.0 * image_features.to(torch.float) @ text_features.to(torch.float).T).softmax(dim=-1).squeeze().cpu().numpy()
#     preds = torch.cat(preds, probs, dim=0)
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

top_10_indices =  np.argmax(preds, axis=1)
# 上位10個のインデックスを表示
print("Top 10 indices:", top_10_indices)
print("Answer" + str(y))
exit()

dir_path = ".\\data\\images\\Images"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

y_ = torch.load("./tensor.pt")

y = torch.load("./data/train_y.pt")
class_y = open("./class_y.txt", "r").read().split('\n')
text = clip.tokenize(class_y).to(device)
idx = [83, 232, 121, 12, 1]


y = y[idx]
y_ = y_[y]
print(y_.shape)
print(y.shape)
print(len(class_y))

with torch.no_grad():
    image_features = y_.to(device).to(torch.float)
    
    text_features = model.encode_text(text).to(torch.float)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # バッチごとに類似度を計算し、インデックスを取得
    top_10_indices_list = []
    for i in range(image_features.shape[0]):
        probs = (100.0 * image_features[i] @ text_features.T).softmax(dim=-1).cpu().numpy()
        top_10_indices = np.argsort(probs)[-10:][::-1]
        top_10_indices_list.append(top_10_indices)
    # # 類似度の計算
    # probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze().cpu().numpy()

# # データをソートし、上位10個のインデックスを取得
# top_10_indices = np.argsort(probs)[-10:][::-1]

# 上位10個のインデックスを表示
print("Top 10 indices:", top_10_indices_list)
print("Answer" + str(y))

exit()
dir_path = ".\\data\\images\\Images"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img = open("./img_path_train.txt", "r").read().split('\n')
with torch.no_grad():
    tensor = torch.empty(0, 512).to(device)
    for img_i in img:
        file_path = dir_path + '/' + img_i
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        tensor = torch.cat((tensor, image_features), dim=0)
print(tensor.shape)
torch.save(tensor, 'tensor1.pt')

# ディレクトリパスとファイルパスの取得
dir_path = "./data/images/Images"
img_paths = open("./img_path_train.txt", "r").read().split('\n')

# バッチサイズの設定
batch_size = 32  # メモリに応じて適宜調整
tensor = torch.empty(0, 512).to(device)


# バッチ処理
for i in range(0, len(img_paths), batch_size):
    batch_paths = img_paths[i:i + batch_size]
    images = []

    for img_path in batch_paths:
        file_path = os.path.join(dir_path, img_path)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        images.append(image)

    images_tensor = torch.cat(images, dim=0)
    
    with torch.no_grad():
        image_features = model.encode_image(images_tensor)
    
    tensor = torch.cat((tensor, image_features), dim=0)

print(tensor.shape)
torch.save(tensor, 'tensor1.pt')
exit()

dir_path = ".\\data\\images\\Images"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img = open("./img_path_train.txt", "r").read().split('\n')
y = torch.load("./data/train_y.pt")
class_y = open("./class_y.txt", "r").read().split('\n')
idx = 123

with torch.no_grad():
    file_path = dir_path + '/' + img[idx]
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    text = clip.tokenize(class_y).to(device)
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # 類似度の計算
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze().cpu().numpy()

# データをソートし、上位10個のインデックスを取得
top_10_indices = np.argsort(probs)[-10:][::-1]

# 上位10個のインデックスを表示
print("Top 10 indices:", top_10_indices)
print("Answer" + str(y[idx]))
exit()

def class_name(name):
     if '/' in name:
          c_name = name.split('/')[0]
     else:
          c_name_list = name.split('_')
          c_name_list.pop()
          c_name = ''
          for i, c_name_i in enumerate(c_name_list):
               if i > 0:
                    c_name = c_name + "_" + c_name_i
               else:
                    c_name = c_name_i
     return c_name

def img_path(name):
     if '/' in name:
          return name
     else:
          c_name_list = name.split('_')
          c_name_list.pop()
          c_name = ''
          for i, c_name_i in enumerate(c_name_list):
               if i > 0:
                    c_name = c_name + "_" + c_name_i
               else:
                    c_name = c_name_i
          c_name = c_name + '/' + name
          return c_name

class_y = [''] * 1854
y = torch.load("./data/train_y.pt")
img = open("./data/train_image_paths.txt", "r").read().split('\n')

dir_path = ".\\data\\images\\Images"

for i, img_i in enumerate(img):
     img[i] = img_path(img_i)

# texts_train = [
# f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
# ]

# print(texts_train == class_y) 
# for i, text in enumerate(texts_train):
#      if text != class_y[i]:
#           print(text + '--' + class_y[i])


# print(len(class_y))  # [1, 2]

# テキストファイルに書き込む
with open("img_path_train.txt", "w") as f:
    for line in img:
        f.write(line + "\n")  # 各要素の後に改行を追加

exit()

dir_path = ".\\data\\images\\Images"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

texts_train = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]
appended_tensor = torch.empty(0, 512).to(device)
for class_name in tqdm(texts_train):
    files = os.listdir(os.path.join(dir_path, class_name))
    image_features_list = torch.empty(0, 512).to(device)
    mean_tensor = torch.empty(0, 512).to(device)
    for file in files:
        with torch.no_grad():
            file_path = os.path.join(dir_path, class_name, file)
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
        image_features_list = torch.cat((image_features_list, image_features), dim=0)
    mean_tensor = torch.mean(image_features_list, dim=0, keepdim=True)
    appended_tensor = torch.cat((appended_tensor, mean_tensor), dim=0)

torch.save(appended_tensor, 'tensor.pt')



for pathname, dirnames, filenames in os.walk(dir_path):
    for filename in filenames:
            image_files_train.append(os.path.join(pathname, filename))


image = preprocess(Image.open(".\\data\\images\\Images\\aardvark\\aardvark_01b.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(texts_train).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(text_features.shape)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # 類似度の計算
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze().cpu().numpy()
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()



from fine_tuning import fine_tuning

fine_tuning()
exit()


import os

dir_path = "./data/images/Images"

files_dir = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]
print(files_dir)

exit()

# .ptファイルのパス
# file_path = "./data/train_y.pt"
file_path = r"C:\Users\asf72\ansel\Document\AI\lecture_DeepLearning\dl_lecture_competition_pub\outputs\2024-07-14\02-21-18\submission.npy"

data = np.load(file_path)
print(data.shape)



# ファイルを読み込む
data = torch.load(file_path)

print(torch.aminmax(data))



# サンプルデータの作成
n_channels = 271
n_times = 281
sfreq = 200  # サンプリング周波数（Hz）
# data = data[0]
from sklearn.preprocessing import StandardScaler
# # RobustScalerの初期化
# scaler = RobustScaler()
# data = data.to('cpu').detach().numpy().astype(np.float64).copy()
# 各サンプル、各チャネルごとにスケーリング
i=0
# チャンネルごとに正規化
scaler = StandardScaler()
shape = data.shape
data = data.view(shape[0], shape[1], -1)
data = scaler.fit_transform(data.numpy().reshape(-1, data.shape[-1])).reshape(shape)
data = torch.from_numpy(data).float()


# # Infoオブジェクトの作成
# info = mne.create_info(
#     ch_names=['MEG' + str(i) for i in range(n_channels)],
#     sfreq=sfreq,
#     ch_types='mag'
# )

# # RawArrayオブジェクトの作成
# raw = mne.io.RawArray(data, info)

# raw.filter(l_freq=1.0, h_freq=None)  # 1 Hzで高域通過フィルタを適用

# # 固定長イベントを生成
# events = mne.make_fixed_length_events(raw, id=1, duration=1)  # 1秒ごとのイベント
# print(events)

# ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
# ica.fit(raw)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw, picks=ica.exclude)

# # データの可視化
# # raw.plot(n_channels=10, duration=1.4, scalings='auto')

# # save as png
# # plt.savefig('X_data.png') # -----(2)
# # プロットを表示
# plt.show()


# # TensorDatasetを作成
# dataset = TensorDataset(data)
# # DataLoaderを作成（バッチサイズを指定）
# batch_size = 1024
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# times = np.arange(281) / 100.0

# all_samples = [] 
# for batch_data in tqdm(dataloader, desc="Train", leave=False):
#     data = batch_data[0].to('cpu').detach().numpy().astype(np.float64).copy()
#     sfreq = 200  # オリジナルのサンプリングレート
#     # data = mne.filter.filter_data(data, sfreq , l_freq=0.1, h_freq=99) # バッチありでも行ける
#     data = mne.filter.resample(data, down=2) # バッチありでもaxis設定すれば行ける
    
#     data = mne.baseline.rescale(data, times, (0, 0.1), verbose=False) # バッチありでも行ける

#     all_samples.append(data)

#     # メモリの開放
#     del batch_data, data
#     gc.collect()

# data = np.concatenate(all_samples, axis=0)



# Infoオブジェクトの作成
info = mne.create_info(
    ch_names=['MEG' + str(i) for i in range(n_channels)],
    sfreq=sfreq,
    ch_types=['mag']*271
)

# RawArrayオブジェクトの作成
raw = mne.io.RawArray(data, info)

# 平均基準化の実行
# raw.set_eeg_reference(ref_channels='average') # MEGデータには必要ない
# data = mne.filter.filter_data(raw, sfreq , l_freq=1, h_freq=40) # してあるから必要ない

# ICAオブジェクトの作成
ica = mne.preprocessing.ICA(n_components=20, random_state=97)  # n_componentsはデータに応じて調整

# ICAの適合
ica.fit(raw)

# ICA成分のプロット（オプション）
ica.plot_components()
plt.show()