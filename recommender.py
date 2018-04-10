# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:49:50 2018

@author: ANURAG
"""

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
import numpy as np

data = fetch_movielens(min_rating=5.0)

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

test_precision = precision_at_k(model, data['test'], k=5).mean()
print(test_precision)

train_precision = precision_at_k(model, data['train'], k=5).mean()
print(train_precision)

def sample_recommendation(model,data,user_ids):
    n_users,n_items=data['train'].shape
    for user_id in user_ids:
        known_positives=data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        scores=model.predict(user_id,np.arange(n_items))
        top_items=data['item_labels'][np.argsort(-scores)]
        print("User %s"%user_id)
        print("     Known positives:")
        
        for x in known_positives[:5]:
            print("            %s"%x)
            
        print("     Recommended:")
        for x in top_items[:5]:
            print("            %s"%x)
            
sample_recommendation(model,data,[3,25,450,10])