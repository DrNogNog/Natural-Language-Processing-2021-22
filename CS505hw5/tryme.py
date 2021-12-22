# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:30:34 2021

@author: Gordo
"""

import csv
import pandas as pd
output = open('xyz.txt','w')
df = pd.read_csv("EditMe.csv")
ages = df["text"]
ages.to_csv(r'C:\Users\Gordo\OneDrive\Desktop\CS505hw5\xyz.txt', header=None, index=None, sep=' ', mode='a')