import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.constants as const
from scipy import stats
import seaborn as sns

plt.rcParams.update({'font.size': 9})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

planet_masses_list = []
for i in range (0,len(datafile)-1):
    if datafile["pl_rade"][i]<1.5:
        m = (0.441 + 0.615*datafile["pl_rade"][i])*(datafile["pl_rade"][i])**3
    elif 1.5<datafile["pl_rade"][i]<4:
        m = 2.69*(datafile["pl_rade"][i])**(0.93)
    else:
        m = 3*(datafile["pl_rade"][i])
    planet_masses_list.append(m)
planet_masses = pd.Series(planet_masses_list)

pd.Series.to_csv(planet_masses, "a.csv")

star_masses = datafile["st_mass"] 
semi_major_axes = datafile["pl_orbsmax"] 

K_list = []
for i in range(0,len(planet_masses)-1):
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        hill_radius = (planet_masses[i] + planet_masses[i+1])/(planet_masses[i] + planet_masses[i+1] + star_masses[i])**(1/3) * ((semi_major_axes[i+1] + semi_major_axes[i])/2)
        K = (semi_major_axes[i+1] - semi_major_axes[i])/hill_radius
        K_list.append(K)
K_data = pd.Series(K_list)
K_data = K_data[K_data > 0]

logK = np.log(K_data)

[mean, std] = stats.norm.fit(logK)
x = np.linspace(np.min(logK), np.max(logK), 1324)

plt.figure(2, figsize=(7.2, 4.5))

plt.subplot(1,2,1)
plt.hist(K_data, density=True, bins=70, histtype="step", color="grey")
plt.xlabel ("K")
plt.ylabel ("PDF(K)")

plt.subplot(1,2,2)
plt.hist(logK, density=True, bins=70, histtype="step", color="grey")
plt.plot(x, stats.norm.pdf(x, mean, std), color=sns.color_palette('Set2')[0])
plt.xlabel ("logK")
plt.show()

