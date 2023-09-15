from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression
from scipy import stats

def get_cindex(Y, P):
    return concordance_index(Y, P)


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


# Prepare for rm2
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


# Prepare for rm2
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)


# Prepare for rm2
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def cos_formula(a, b, c):
    ''' formula to calculate the angle between two edges
        a and b are the edge lengths, c is the angle length.
    '''
    res = (a**2 + b**2 - c**2) / (2 * a * b)
    # sanity check
    res = -1. if res < -1. else res
    res = 1. if res > 1. else res
    return np.arccos(res)

def get_rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def get_mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def get_sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def get_pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def get_spearman(y,f):
    sp = stats.spearmanr(y,f)[0]

    return sp

def evaluate_reg(Y, F):
    not_nan_indices = ~np.isnan(Y)
    Y = Y[not_nan_indices]
    F = F[not_nan_indices]
    
    return { 
        'mse': float(get_mse(Y,F)),
        'rmse': float(get_rmse(Y,F)),
        'mae': float(get_mae(Y,F)),
        'sd': float(get_sd(Y,F)),
        'pearson': float(get_pearson(Y,F)),
        'spearman': float(get_spearman(Y,F)),
        'rm2': float(get_rm2(Y,F)),
        'ci': float(get_cindex(Y,F))
    }

def evaluate_cls(Y,P,threshold=0.5):
    predicted_label = P > threshold
    
    return {
        'roc': float(roc_auc_score(Y,P)),
        'prc': float(average_precision_score(Y,P)),
        'f1': float(f1_score(Y,predicted_label)),
        'recall':float(recall_score(Y, predicted_label)),
        'precision': float(precision_score(Y, predicted_label))
    }

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score

def multiclass_ap(Y_test, y_score, n_classes):
    # For each class
    # precision = dict()
    # recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        num_labels = Y_test[:, i].sum()
        if num_labels == 0: continue
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    return sum(average_precision.values()) / len(average_precision)

def evaluate_mcls(Y,P):
    nclass = P.shape[-1]
    # Filter Y and P based on n_classes
    valid_indices = np.isin(Y, np.arange(nclass))
    Y = Y[valid_indices]
    P = P[valid_indices]
    
    onehot_y = indices_to_one_hot(Y,nclass)
    try:
        roc = roc_auc_score(onehot_y, P, average='macro',multi_class='ovo')
        prc = multiclass_ap(onehot_y, P, n_classes=nclass)
    except:
        roc = -999
        prc = -999
    pred_class = np.argmax(P,axis=-1)
    acc = accuracy_score(Y,pred_class)
    multi_result = {
        'multiclass_roc': float(roc),
        'multiclass_prc': float(prc),
        'multiclass_accuracy':float(acc),
        'macro_f1':float(f1_score(Y,pred_class,average='macro')),
    }

    return multi_result 


if __name__ == '__main__':
    G = [5.0, 7.251812, 5.0, 7.2676063, 5.0, 8.2218485, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.7212462, 5.0, 5.0, 5.0, 5.0, 5.0, 6.4436975, 5.0, 5.0, 5.60206, 5.0, 5.0, 5.1426673, 5.0, 5.0, 6.387216, 5.0, 5.0, 5.0, 6.251812, 5.0, 5.0, 5.0, 5.0, 5.0, 6.958607, 5.0, 5.0, 5.0, 5.0, 7.1739254, 5.0, 5.0, 5.0, 6.207608, 5.0, 5.5850267, 5.0, 6.481486, 5.0, 6.455932, 5.0, 5.0, 6.853872, 5.7212462, 5.0, 5.6575775, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.29243, 5.6382723, 5.0, 5.0, 5.0, 5.0, 5.0, 5.4317985, 5.0, 6.6777806, 5.0, 5.0, 5.0, 5.0, 5.5086384, 5.0, 5.0, 5.4436975, 5.0, 5.0, 5.6777806, 5.0, 5.075721, 5.0, 5.0, 8.327902, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    P = [5.022873, 7.0781856, 4.9978094, 6.7880363, 5.0082135, 8.301622, 5.199977, 5.031757, 5.282739, 5.1505866, 5.0371256, 5.0158253, 7.235809, 5.0488424, 5.0158954, 5.014982, 5.0353045, 5.0385847, 6.210839, 5.0246162, 5.040341, 5.9972534, 5.022253, 5.024069, 5.0325136, 5.858346, 5.1466026, 7.353938, 5.041976, 5.010902, 5.0101852, 5.7545958, 5.0263815, 5.0000725, 4.985109, 5.055313, 5.0001907, 6.8203254, 5.0954485, 5.1212735, 5.0224247, 5.0497823, 6.8255396, 5.0044026, 4.9908457, 5.0110598, 6.855809, 5.297818, 6.2044125, 5.0267057, 6.1194935, 5.005172, 5.6843953, 5.0014734, 5.0232143, 7.3333316, 5.8368444, 5.2844615, 5.8721313, 5.040511, 5.057362, 5.0058765, 5.018214, 5.0278683, 4.995488, 6.170251, 5.2143936, 5.0082054, 5.0141716, 5.560684, 5.0162783, 5.022541, 5.4540567, 5.023486, 5.0640993, 4.9965744, 5.0399494, 5.0136223, 5.1999803, 6.3908367, 5.022854, 5.0350113, 5.002722, 5.0313835, 5.175599, 5.1362724, 5.137325, 5.6480265, 5.03323, 5.054763, 8.333924, 5.0164843, 5.2512374, 5.02013, 5.023677, 5.0309353, 5.031672, 6.3660593, 5.035504, 5.0222054]
    # print('regression result:', evaluate_reg(np.array(G), np.array(P)))
    # print('cls result:',evaluate_cls(np.array(G) >= 6, np.array(P),6))

    t = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 2, 2, 2, 2])
    p = np.array([[ 0.8746, -0.9049,  0.4639],
        [ 0.8708, -0.8453,  0.4591],
        [ 0.8843, -0.9211,  0.5301],
        [ 0.7957, -0.9277,  0.5806],
        [ 0.8791, -0.8414,  0.4515],
        [-0.0475, -0.2103,  0.5898],
        [-0.0173, -0.0968,  0.4454],
        [ 0.8182, -1.0273,  0.6466],
        [ 0.9068, -0.9533,  0.5292],
        [-0.8911,  1.6034, -0.4516],
        [-1.0852,  1.1397,  0.3087],
        [-0.6816,  0.7026,  0.3080],
        [ 0.3641, -0.2075,  0.2326],
        [ 0.5106, -0.1516,  0.0526],
        [-0.2555, -0.6679,  1.3138],
        [ 0.0850,  0.0502,  0.1387],
        [-0.6787,  0.4709,  0.4442],
        [-0.5439,  0.4989,  0.2730],
        [-0.1568, -1.2820,  1.9857],
        [-0.0165, -1.2909,  1.7805]])
    print(evaluate_mcls(t,p))

    