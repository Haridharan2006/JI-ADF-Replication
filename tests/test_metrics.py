import numpy as np

from utils.metrics import compute_metrics

labels=np.array([0,1,2,1])

pred=np.array([0,1,1,1])

metrics=compute_metrics(labels,pred)

print(metrics)