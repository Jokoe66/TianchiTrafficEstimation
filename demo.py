from datetime import datetime
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import lightgbm
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold

def get_data(df, img_path):
    map_id_list=[]
    label=[]
    key_frame_list=[]
    jpg_name_1=[]
    jpg_name_2=[]
    gap_time_1=[]
    gap_time_2=[]
    im_diff_mean=[]
    im_diff_std=[]
    feats = defaultdict(
        lambda :defaultdict(
            lambda : []))
    
    for idx in range(len(df)):
        s = df.iloc[idx]
        map_id=s["id"]
        map_key=s["key_frame"]
        frames=s["frames"]
        status=s["status"]
        key = [frame['frame_name'] for frame in frames].index(map_key)
        for name, feat in s['feats'].items():
            feats[name]['mean'].append(np.mean(feat))
            feats[name]['std'].append(np.std(feat))
            feats[name]['key'].append(feat[key])
            feats[name]['gap'].append(np.max(feat) - np.min(feat))
        
        for i in range(0,len(frames)-1):
            f=frames[i]
            f_next=frames[i+1]

            map_id_list.append(map_id)
            key_frame_list.append(map_key)
            jpg_name_1.append(f["frame_name"])
            jpg_name_2.append(f_next["frame_name"])
            gap_time_1.append(f["gps_time"])
            gap_time_2.append(f_next["gps_time"])
            label.append(status)
    train_df= pd.DataFrame({
        "map_id":map_id_list,
        "label":label,
        "key_frame":key_frame_list,
        "jpg_name_1":jpg_name_1,
        "jpg_name_2":jpg_name_2,
        "gap_time_1":gap_time_1,
        "gap_time_2":gap_time_2,
    })

    train_df["gap"]=train_df["gap_time_2"]-train_df["gap_time_1"]
    train_df["gap_time_today"]=train_df["gap_time_1"]%(24*3600)
    train_df["hour"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).hour)
    train_df["minute"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).minute)
    train_df["day"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).day)
    train_df["dayofweek"]=train_df["gap_time_1"].apply(
        lambda x:datetime.fromtimestamp(x).weekday())
    
    train_df["key_frame"]=train_df["key_frame"].apply(lambda x:int(x.split(".")[0]))
    
    train_df=train_df.groupby("map_id").agg({"gap":["mean","std"],
                                             "hour":["mean"],
                                             "minute":["mean"],
                                             "dayofweek":["mean"],
                                             "gap_time_today":["mean","std"],
                                             "label": ["mean"],
                                            }).reset_index()
    train_df.columns=["map_id","gap_mean","gap_std",
                      "hour_mean","minute_mean","dayofweek_mean",
                      "gap_time_today_mean","gap_time_today_std",
                      "label"]
    train_df["label"]=train_df["label"].apply(int)
    train_df = pd.concat([
        train_df,
        *[pd.Series(v, name=f'{name}_{k}') for name, feat in feats.items() for k, v in feat.items()],
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
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=50,
                                  #feval=acc_score_vali,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                print("\n".join(("%s: %.2f" % x) for x in
                                list(sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
                                       reverse=True))[:200]
                                ))
                pre = model.predict(te_x, num_iteration=model.best_iteration)
                pred = model.predict(test_x, num_iteration=model.best_iteration)
                train[test_index] = pre
                test_pre[i, :] = pred
                cv_scores.append(log_loss(te_y, pre))
                
                f1_list=f1_score(te_y,np.argmax(pre,axis=1),average=None)
                f1s.append(f1_list)
                f1=0.2*f1_list[0]+0.2*f1_list[1]+0.6*f1_list[2]
                
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
    return train, test, test_pre_all, np.mean(f1_scores)


def lgb(x_train, y_train, x_valid, weights):
    lgb_train, lgb_test, sb, cv_scores = stacking(lightgbm, x_train, y_train, x_valid, "lgb", 3, weights)
    return lgb_train, lgb_test, sb, cv_scores


if __name__ == '__main__':
    train_json = pd.read_json("data/enriched_annotations_train.json")
    test_json = pd.read_json("data/enriched_annotations_test.json")


    train_df = get_data(train_json[:], "data/amap_traffic_train_0712")
    weights = np.array([1.0, 5.0, 2.0])
    weights /= np.sum(weights)
    weights *= 3 * 1.5
    weights = pd.Series(train_df['label'].apply(lambda x: weights[int(x)]), name='weight')
    test_df = get_data(test_json[:], "data/amap_traffic_test_0712")
    
    select_features=["gap_mean","gap_std",
#                      "hour_mean", "minute_mean","dayofweek_mean",
                     "gap_time_today_mean","gap_time_today_std",
                     "closest_vehicle_distance_mean",
                     "closest_vehicle_distance_std",
                     "closest_vehicle_distance_key",
#                      "closest_vehicle_distance_gap",
                     "main_lane_vehicles_mean",
#                      "main_lane_vehicles_std",
#                      "main_lane_vehicles_key",
#                      "main_lane_vehicles_gap",
                     "total_vehicles_mean",
                     "total_vehicles_std",
#                      "total_vehicles_key",
#                      "total_vehicles_gap",
#                      "lanes_mean",
#                      "lanes_std",
#                      "lanes_key",
                     "lane_length_mean",
                     "lane_length_std",
                     "lane_length_key",
#                      "lane_length_gap",
                     "lane_width_mean",
                     "lane_width_std",
#                      "lane_width_key",
#                      "lane_width_gap",
#                      "vehicle_distances_mean_mean",
#                      "vehicle_distances_mean_std",
#                      "vehicle_distances_mean_key",
                     "vehicle_distances_std_mean",
#                      "vehicle_distances_std_std",
                     "vehicle_distances_std_key",
                     "vehicle_area_mean",
                     "vehicle_area_std",
                     "vehicle_area_key",
                     "vehicle_area_gap",
                    ]

    train_x=train_df[select_features].copy()
    train_y=train_df["label"]
    valid_x=test_df[select_features].copy()

    lgb_train, lgb_test, sb, m = lgb(train_x, train_y, valid_x, weights)
    
    # submit
    sub=test_df[["map_id"]].copy()
    sub["pred"]=np.argmax(lgb_test,axis=1)

    result_dic=dict(zip(sub["map_id"],sub["pred"]))
    with open("data/amap_traffic_annotations_test.json", "r") as f:
        content=f.read()
    content=json.loads(content)
    for i in content["annotations"]:
        i['status']=result_dic[int(i["id"])]
    with open(f"sub_{m:.4f}.json","w") as f:
        f.write(json.dumps(content))
