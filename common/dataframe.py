import copy
import pandas as pd

def sort_optimal_experiments(optimal, asc, des):
    ascending = []
    ascending.extend([True for _ in asc])
    ascending.extend([False for _ in des])
    target = []
    target.extend([item for item in asc])
    target.extend([item for item in des])
    
    
    use_id = []
    for id, item in enumerate(optimal):
        if type(item['metrics']) == dict:
            use_id.append(id)
    
    if len(use_id) == 0:
        return optimal[0]
    
    if len(use_id) == 1:
        return optimal[use_id[0]]
    
    metric_data = []
    for id in use_id:
        metric_data.append(optimal[id]['metrics'])    
    
    df = pd.DataFrame.from_dict(metric_data).fillna(0.)
    df[target] = df[target].astype('float')
    df = df.sort_values(target, axis=0, ascending=ascending)     
    
    return optimal[use_id[df.index[0]]]