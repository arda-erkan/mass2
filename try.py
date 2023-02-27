import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.constants as const

plt.rcParams.update({'font.size': 9})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

planet_masses = datafile["pl_bmasse"] 
star_masses = datafile["st_mass"] 
semi_major_axes = datafile["pl_orbsmax"] 

M_total_system_list = []
hill_list = []
for i in range(0,len(datafile)-1):
    M = 0
    for r in datafile["hostname"]:
        index = 0
        if r == datafile["hostname"][i]:
            M += datafile["pl_bmasse"][index]
        index += 1
        print(M)