import numpy as np
from random import random
from sklearn.model_selection import train_test_split

#array([[0.1, 0.2], [0.2, 0.2]])
#array([[0.3], [0.4]])

def generate_dataset(num_samples, test_size):
    x = np.array([random()/2 for _ in range(2)] for _ in range(2000))
    y = np.array([[i[0]+ i[1]]for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

#build model
#compile model
#train model
#evaluate model
#make predictions
