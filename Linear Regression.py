import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def estimate_coef_linear_regression(x,y):
    # y = B_0 + B_1*x
    n = np.size(x)

    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y*x) - n*m_x*m_y
    SS_xx = np.sum(x*x) - n(m_x)**2

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)

data = pd.read_csv('D:/NguyenTienDat_23520262/Nam 2,5/Hoc may thong ke/forest+fires/forestfires.csv')

b_0_predicted, b_1_predicted = estimate_coef_linear_regression(data['X'], data['Y'])

# print(b_0_predicted, b_1_predicted)
