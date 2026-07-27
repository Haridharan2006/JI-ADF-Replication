import numpy as np

from utils.visualization import *

plot_loss(

[1,0.9,0.8],

[1.2,1.0,0.95]

)

plot_accuracy(

[50,60,70],

[48,59,68]

)

cm=np.array(

[[10,2],[1,8]]

)

plot_confusion_matrix(

cm,

["A","B"]

)