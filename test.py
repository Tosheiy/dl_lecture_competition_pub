import tensorflow as tf

def custom_loss(y_true, y_pred):
    t = 0.5
    lamda = 0.8
    
    # 正規化
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)

    # ペアワイズコサイン類似度
    cosine_similarity_matrix = tf.matmul(y_pred_norm, y_true_norm, transpose_b=True)

    LossClip = tf.reduce_sum(cosine_similarity_matrix, axis=-1) 
    LossMse =0
    
    # expを取ってスコアを計算
    exp_scores = tf.exp(cosine_similarity_matrix / t)

    # 対角要素は除去（対角の部分には相関があるので、損失には不要なケースがある）
    exp_scores -= tf.linalg.diag(tf.linalg.diag_part(exp_scores))

    # 対角要素を除いて各行ごとの合計
    score_pred_l = tf.reduce_sum(exp_scores, axis=1)
    score_pred_r = tf.reduce_sum(exp_scores, axis=0)
    
    # 損失を計算
    LossClip = -1 * tf.reduce_mean(tf.math.log(score_pred_l) + tf.math.log(score_pred_r))
 
    # テンソルの差を計算
    difference = y_true - y_pred
    # L2ノルムの平方を計算
    LossMse = tf.reduce_mean(tf.square(difference)) / 512
     
    return LossClip * lamda +  LossMse * (1-lamda)