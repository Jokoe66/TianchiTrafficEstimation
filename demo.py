from datetime import datetime
import json
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import lightgbm
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold

def get_data(df):
    feats = defaultdict(lambda :defaultdict(list))
    dnn_feats = defaultdict(list)
    
    for idx in range(len(df)):
        s = df.iloc[idx]
        map_key = s["key_frame"]
        frames = s["frames"]
        frames.sort(key=lambda x: x['frame_name'])
        key = [frame['frame_name'] for frame in frames].index(map_key)
        dfeats = s['feats']['dnn_feats']
        if dfeats is not None:
            for j, dfeat in enumerate(dfeats):
                dnn_feats[f'dnn_feats_{j}'].append(dfeat)
        for name, feat in s['feats'].items():
            if name == 'dnn_feats': continue
            feats[name]['mean'].append(np.mean(feat))
            feats[name]['std'].append(np.std(feat))
            feats[name]['key'].append(feat[key])
            feats[name]['gap'].append(np.max(feat) - np.min(feat))

    train_df = pd.concat([
        df['id'],
        df['status'].rename('label').apply(int),
        *([pd.Series(v, name=f'{name}_{k}') for name, feat in feats.items()
            for k, v in feat.items()]
          + [pd.Series(feat, name=name) for name, feat in dnn_feats.items()]
          )
    ], axis=1)
    
    return train_df


def stacking(clf, train_x, train_y, test_x, clf_name, class_num=1, weights=None):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 10
    seed = 2029
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))
    test_pre = np.zeros((folds, test_x.shape[0], class_num))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    f1_scores = []
    f1s = []
    cv_rounds = []
    feature_importance = pd.DataFrame()

    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        weight = weights[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            train_matrix.set_weight(weight)
            test_matrix = clf.Dataset(te_x, label=te_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                #'metric': 'None',
                'is_unbalance': False,
                'metric': 'multi_logloss',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 3 - 1,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'feature_fraction_bynode': 1.0,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'min_data_in_leaf': 20,
                'learning_rate': 0.05,
                'seed': seed,
                'nthread': 28,
                'num_class': class_num,
                'silent': True,
                'verbose': -1,
            }

            num_round = 4000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round,
                        valid_sets=test_matrix, verbose_eval=50,
                        #feval=acc_score_vali,
                        early_stopping_rounds=early_stopping_rounds)
                feature_importance = feature_importance.append(pd.DataFrame(
                    {'importance': model.feature_importance("gain")}, index=predictors))
                pre = model.predict(te_x, num_iteration=model.best_iteration)
                pred = model.predict(test_x, num_iteration=model.best_iteration)
                train[test_index] = pre
                test_pre[i, :] = pred
                cv_scores.append(log_loss(te_y, pre))
                
                f1_list = f1_score(te_y,np.argmax(pre,axis=1),average=None)
                f1s.append(f1_list)
                f1 = 0.1 * f1_list[0] + 0.2 * f1_list[1] + 0.3 * f1_list[2] + 0.4 * f1_list[3]
                
                f1_scores.append(f1)
                cv_rounds.append(model.best_iteration)
                test_pre_all[i, :] = np.argmax(pred, axis=1)

        print("%s now score is:" % clf_name, cv_scores)
        print("%s now f1-score is:" % clf_name, f1_scores)
        print("%s now round is:" % clf_name, cv_rounds)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores), np.mean(f1_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    print(f"f1_scores_mean: {np.mean(f1s, axis=0)}")
    feature_importance = feature_importance.groupby(level=0).mean().sort_values(
        by='importance', ascending=False)
    print(feature_importance)
    features = feature_importance.index.tolist()
    return train, test, test_pre_all, np.mean(f1_scores), features


def lgb(x_train, y_train, x_valid, weights):
    lgb_train, lgb_test, sb, cv_scores, features = stacking(
        lightgbm, x_train, y_train, x_valid, "lgb", 4, weights)
    
    # select important features
    important_features = features[:int(len(features) * 0.7)]
    x_train = x_train[important_features]
    x_valid = x_valid[important_features]
    lgb_train, lgb_test, sb, cv_scores, features = stacking(
        lightgbm, x_train, y_train, x_valid, "lgb", 4, weights)

    return lgb_train, lgb_test, sb, cv_scores


if __name__ == '__main__':
    data_root = '../data'
    user_data_root = '../user_data'
    train_json = pd.DataFrame(pd.read_pickle(os.path.join(
        user_data_root, "enriched_annotations_train_final1_dnn.pkl")))
    test_json = train_json
    # uncomment before submitting
    test_json = pd.DataFrame(pd.read_pickle(os.path.join(
        user_data_root, "enriched_annotations_test_final_dnn.pkl")))

    train_df = get_data(train_json[:])
    weights = np.array([0.6, 5.0, 2.0, 0.4])
    weights /= np.sum(weights)
    weights *= 3 * 1.5
    weights = pd.Series(train_df['label'].apply(lambda x: weights[int(x)]), name='weight')
    test_df = get_data(test_json[:])
    
    select_features=["closest_vehicle_distance_mean",
                     "closest_vehicle_distance_std",
                     "closest_vehicle_distance_key",
                     "closest_vehicle_distance_gap",
                     "main_lane_vehicles_mean",
                     "main_lane_vehicles_std",
                     "main_lane_vehicles_key",
                     "main_lane_vehicles_gap",
                     "total_vehicles_mean",
                     "total_vehicles_std",
                     "total_vehicles_key",
                     "total_vehicles_gap",
                     "lanes_mean",
                     "lanes_std",
                     "lanes_key",
                     "lane_length_mean",
                     "lane_length_std",
                     "lane_length_key",
                     "lane_length_gap",
                     "lane_width_mean",
                     "lane_width_std",
                     "lane_width_key",
                     "lane_width_gap",
                     "vehicle_distances_mean_mean",
                     "vehicle_distances_mean_std",
                     "vehicle_distances_mean_key",
                     "vehicle_distances_std_mean",
                     "vehicle_distances_std_std",
                     "vehicle_distances_std_key",
                     "vehicle_area_mean",
                     "vehicle_area_std",
                     "vehicle_area_key",
                     "vehicle_area_gap",
                     "num_obstacles_mean",
                     "num_obstacles_std",
                     "num_obstacles_key",
                     "vehicle_depth_mean_mean",
                     "vehicle_depth_mean_std",
                     "vehicle_depth_mean_key",
                     "vehicle_depth_std_mean",
                     "vehicle_depth_std_std",
                     "vehicle_depth_std_key",
                     "closest_vehicle_depth_mean",
                     "closest_vehicle_depth_std",
                     "closest_vehicle_depth_key",
                     "closest_vehicle_depth_gap",
                    ]
    select_features += [f"dnn_feats_{j}" for j in range(128)] # dnn_feats dim

    train_x = train_df[select_features].copy()
    train_y = train_df['label']
    valid_x = test_df[select_features].copy()

    lgb_train, lgb_test, sb, m = lgb(train_x, train_y, valid_x, weights)
    
    # submit
    # uncomment before submmiting
    sub=test_df[["id"]].copy()
    sub["pred"]=np.argmax(lgb_test,axis=1)

    result_dic=dict(zip(sub["id"], sub["pred"]))
    with open(
        os.path.join("/tcdata/amap_traffic_final_test_0906.json"),
        "r"
        ) as f:
        content = f.read()
    content = json.loads(content)
    for i in content["annotations"]:
        i['status'] = result_dic[i["id"]]
    with open(f"result.json","w") as f:
        f.write(json.dumps(content))
