


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
plt.rcParams["figure.figsize"] = (10, 5)


curated_df_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/main/datasets/malaria_project_curated_data.csv"
malaria_df = pd.read_csv(curated_df_url)
malaria_df = malaria_df.set_index('Year')
malaria_df.head()


#Generating dummy variables for regions
subset_2013 = malaria_df.loc["2013-01-01"].copy()
subset_2013["Is Asia & Pacific"] = pd.get_dummies(subset_2013["Region"])["Asia & Pacific"]
subset_2013["Is Arab States"] = pd.get_dummies(subset_2013["Region"])["Arab States"]
subset_2013["Is Africa"] = pd.get_dummies(subset_2013["Region"])["Africa"]
subset_2013["Is South/Latin America"] = pd.get_dummies(subset_2013["Region"])["South/Latin America"]
subset_2013["Is Europe"] = pd.get_dummies(subset_2013["Region"])["Europe"]
subset_2013["Is Middle east"] = pd.get_dummies(subset_2013["Region"])["Middle east"]


incidence_model_vars = ["Is Asia & Pacific", "Is Arab States", "Is Africa",
                        "Is South/Latin America", "Is Europe", "Is Middle east",
                        "AverageTemperature", "GDPpcPPP"]

subset_2013 = subset_2013.dropna(subset = incidence_model_vars).copy() # making sure there are no NA values

incidence_model_2013 = LinearRegression()
incidence_model_2013.fit(subset_2013[incidence_model_vars], subset_2013["Incidence"])
incidence_model_2013_coeff = incidence_model_2013.coef_[:]


for i in range(len(incidence_model_2013_coeff)):
    print('For', incidence_model_vars[i], 'variable, the regression coefficient is: {:.2f}'.format(incidence_model_2013_coeff[i]))


malaria_df_dropna = malaria_df.copy()
malaria_df_dropna["Is Asia & Pacific"] = pd.get_dummies(malaria_df_dropna["Region"])["Asia & Pacific"]
malaria_df_dropna["Is Arab States"] = pd.get_dummies(malaria_df_dropna["Region"])["Arab States"]
malaria_df_dropna["Is Africa"] = pd.get_dummies(malaria_df_dropna["Region"])["Africa"]
malaria_df_dropna["Is South/Latin America"] = pd.get_dummies(malaria_df_dropna["Region"])["South/Latin America"]
malaria_df_dropna["Is Europe"] = pd.get_dummies(malaria_df_dropna["Region"])["Europe"]
malaria_df_dropna["Is Middle east"] = pd.get_dummies(malaria_df_dropna["Region"])["Middle east"]
malaria_df_dropna = malaria_df_dropna.dropna(subset=incidence_model_vars).copy() # making sure there are no NA values

incidence_model_pooled = LinearRegression()
incidence_model_pooled.fit(malaria_df_dropna[incidence_model_vars], malaria_df_dropna["Incidence"])
incidence_model_pooled_coeff = incidence_model_pooled.coef_[:]


for i in range(len(incidence_model_pooled_coeff)):
    print('For', incidence_model_vars[i], 'variable, the regression coefficient is: {:.2f}'.format(incidence_model_pooled_coeff[i]))


















