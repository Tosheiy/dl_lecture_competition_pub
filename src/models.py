import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 512
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X) + X
        X = F.gelu(self.batchnorm2(X))

        return self.dropout(X)

# class ShallowConvNet(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         seq_len: int,
#         in_channels: int,
#         hid_dim: int = 512
#     ) -> None:
#         super().__init__()

#         self.blocks = nn.Sequential(
#             ConvBlock(in_channels, hid_dim),
#             ConvBlock(hid_dim, hid_dim),
#         )

#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             Rearrange("b d 1 -> b d"),
#             nn.Linear(hid_dim, num_classes),
#         )

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """_summary_
#         Args:
#             X ( b, c, t ): _description_
#         Returns:
#             X ( b, num_classes ): _description_
#         """
#         X = self.blocks(X)

#         return self.head(X)


# class ConvBlock(nn.Module):
#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         kernel_size: int = 3,
#         p_drop: float = 0.1,
#     ) -> None:
#         super().__init__()
        
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#         self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
#         self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
#         self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
#         self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
#         self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
#         self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

#         self.dropout = nn.Dropout(p_drop)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         if self.in_dim == self.out_dim:
#             X = self.conv0(X) + X  # skip connection
#         else:
#             X = self.conv0(X)

#         X = F.gelu(self.batchnorm0(X))

#         X = self.conv1(X) + X  # skip connection
#         X = F.gelu(self.batchnorm1(X))

#         X = self.conv2(X) + X
#         X = F.gelu(self.batchnorm2(X))

#         return self.dropout(X)

    
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape
from keras.constraints import max_norm

def EEGNet(output, Channels = 271, Time_seq = 281, 
             dropoutRate = 0.25, kernLength = 100, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ 
    EEGNet - a compact CNN model for EEG-based neurodecoding
    """
    # # 入力層の形状を (Channels, Time_seq) に設定
    # input1 = Input(shape=(Channels, Time_seq))
    # # リシェイプ層 (1, チャネル数, 時間データ数) にリシェイプ
    # reshaped = Reshape((1, Channels, Time_seq))(input1)
    input1 = Input(shape=(Channels, Time_seq, 1))
    block1 = Conv2D(F1, (1, kernLength), padding = 'same', input_shape = (1, Channels, Time_seq), use_bias = False)(input1)
    block1 = BatchNormalization(axis = -1)(block1)
    block1 = DepthwiseConv2D((Channels, 1), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm(1))(block1)
    block1 = BatchNormalization(axis = -1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same')(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    block2 = Flatten(name = 'flatten')(block2)
    
    dense  = Dense(output, name = 'dense', kernel_constraint = max_norm(norm_rate))(block2)
    # dense1 = Dense(1024, activation='elu', kernel_constraint=max_norm(norm_rate))(block2)
    # dense1 = Dropout(dropoutRate)(dense1)
    # dense = Dense(output, activation='linear', kernel_constraint=max_norm(norm_rate))(dense1)
    
    return Model(inputs=input1, outputs=dense)


from tensorflow.keras.losses import Loss
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package='Custom', name='custom_loss')
def custom_loss(y_true, y_pred):
    t = 0.5
    lamda = 0.8

    # 正規化
    y_true_normalize = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_normalize = tf.nn.l2_normalize(y_pred, axis=-1)

    # ペアワイズコサイン類似度（サイズは正規化しているので１）
    cosine_similarity_matrix = tf.matmul(y_pred_normalize, y_true_normalize, transpose_b=True)
    
    # expを取ってスコアを計算
    exp_scores = tf.exp(cosine_similarity_matrix / t)

    # 対角成分を抽出
    score_pred_molecule = tf.linalg.diag_part(exp_scores)
    
    # 分母の計算
    score_pred_l = tf.reduce_sum(exp_scores, axis=-1)
    score_pred_r = tf.reduce_sum(exp_scores, axis=0)
    score_pred_r = tf.transpose(score_pred_r)
    frac_l = tf.divide(score_pred_molecule, score_pred_l)
    frac_r = tf.divide(score_pred_molecule, score_pred_r)

    # 損失を計算
    LossClip = -1 * tf.reduce_mean(tf.math.log(frac_l) + tf.math.log(frac_r))

    mse = tf.keras.losses.MeanSquaredError()
    LossMse = mse(y_true, y_pred)
     
    return lamda * LossClip + (1 - lamda) * LossMse