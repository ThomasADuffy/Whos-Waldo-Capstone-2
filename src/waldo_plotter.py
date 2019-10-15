import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WaldoPlotter():
    '''This is a plotting function'''

    def __init__(self,df,score):
        self.df = df
        self.score_loss = score[0]
        self.score_accuracy = score[1]