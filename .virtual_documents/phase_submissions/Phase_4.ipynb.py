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
malaria_df = malaria_df.set_index("Year")
malaria_df.head()


by_year = malaria_df.groupby('Year')
year_2013 = malaria_df.loc['2013-01-01']


plt.plot(by_year['Confirmed Cases'].mean(), label='Mean')
plt.plot(by_year['Confirmed Cases'].median(), label='Median')
plt.title('Global Malaria Confirmed Cases Over Time (2000-2013)')
plt.xlabel('Year')
plt.ylabel('No. of Confirmed Cases')
plt.legend()
plt.show()


plt.plot(by_year['Confirmed Cases'].median(), color='orange')
plt.title('Median Global Malaria Confirmed Cases Over Time (2000-2013)')
plt.xlabel('Year')
plt.ylabel('No. of Confirmed Cases')
plt.show()


sns.lineplot(x='Year', y='Confirmed Cases', hue='Region', data=malaria_df)
plt.title('Global Malaria Confirmed Cases Over Time per Region (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average No. of Confirmed Cases (95% CI)')
plt.legend()
plt.show()


sns.lineplot(x='Year', y='Confirmed Cases', hue='Region', data=malaria_df[malaria_df['Region']get_ipython().getoutput("='Africa'])")
plt.title('Global Malaria Confirmed Cases Over Time per Region Excluding Africa (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average No. of Confirmed Cases (95% CI)')
plt.legend()
plt.show()


plt.plot(by_year['Incidence'].mean(), label='Mean')
plt.plot(by_year['Incidence'].median(), label='Median')
plt.title('Global Malaria Incidence Over Time (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Incidence (per 1,000 population)')
plt.legend()
plt.show()


plt.plot(by_year['Incidence'].median(), color='orange')
plt.title('Median Global Malaria Incidence Over Time (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Incidence (per 1,000 population)')
plt.show()


sns.lineplot(x='Year', y='Incidence', hue='Region', data=malaria_df)
plt.title('Global Malaria Incidence Over Time per Region (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average Incidence (per 1,000 population) (95% CI)')
plt.legend()
plt.show()


sns.lineplot(x='Year', y='Incidence', hue='Region', data=malaria_df[malaria_df['Region']get_ipython().getoutput("='Africa'])")
plt.title('Global Malaria Incidence Over Time per Region Excluding Africa (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average Incidence (per 1,000 population) (95% CI)')
plt.legend()
plt.show()


plt.plot(by_year['Deaths'].mean(), label='Mean')
plt.plot(by_year['Deaths'].median(), label='Median')
plt.title('Global Malaria Deaths Over Time (2000-2013)')
plt.xlabel('Year')
plt.ylabel('No. of Confirmed Deaths')
plt.legend()
plt.show()


plt.plot(by_year['Deaths'].median(), color='orange')
plt.title('Median Global Malaria Deaths Over Time (2000-2013)')
plt.xlabel('Year')
plt.ylabel('No. of Confirmed Deaths')
plt.show()


sns.lineplot(x='Year', y='Deaths', hue='Region', data=malaria_df)
plt.title('Global Malaria Confirmed Deaths Over Time per Region (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average No. of Confirmed Deaths (95% CI)')
plt.legend()
plt.show()


sns.lineplot(x='Year', y='Deaths', hue='Region', data=malaria_df[malaria_df['Region']get_ipython().getoutput("='Africa'])")
plt.title('Global Malaria Confirmed Deaths Over Time per Region Excluding Africa (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average No. of Confirmed Deaths (95% CI)')
plt.legend()
plt.show()


plt.hist(year_2013['Incidence'], bins=30)
plt.title('Distribution of Malaria Incidence (2013)')
plt.xlabel('Incidence (per 1,000 people)')
plt.ylabel('Count')
plt.show()


sns.boxplot(x='Region', y='Incidence', data=year_2013)
sns.stripplot(x='Region', y='Incidence', data=year_2013, color='black', alpha=0.6)
plt.title('Incidence per Global Region (2013)')
plt.xlabel('Global Region')
plt.ylabel('Incidence (per 1,000 population)')
plt.show()


sns.boxplot(x='Global South', y='Incidence', data=year_2013)
sns.stripplot(x='Global South', y='Incidence', data=year_2013, color='black', alpha=0.6)
plt.title('Incidence per Development Status (2013)')
plt.xlabel('Development Status')
plt.ylabel('Incidence (per 1,000 population)')
plt.show()


plt.hist(year_2013['Deaths'], bins=30)
plt.title('Distribution of Malaria Deaths (2013)')
plt.xlabel('No. of Deaths')
plt.ylabel('Count')
plt.show()


sns.boxplot(x='Region', y='Deaths', data=year_2013)
sns.stripplot(x='Region', y='Deaths', data=year_2013, color='black', alpha=0.6)
plt.title('Confirmed Deaths per Global Region (2013)')
plt.xlabel('Global Region')
plt.ylabel('Confirmed Deaths')
plt.show()


sns.boxplot(x='Region', y='Deaths', data=year_2013[year_2013['Region']get_ipython().getoutput("='Africa'])")
sns.stripplot(x='Region', y='Deaths', data=year_2013[year_2013['Region']get_ipython().getoutput("='Africa'], color='black', alpha=0.6)")
plt.title('Confirmed Deaths per Global Region excluding Africa (2013)')
plt.xlabel('Global Region')
plt.ylabel('No. of Confirmed Deaths')
plt.show()


sns.boxplot(x='Global South', y='Deaths', data=year_2013)
sns.stripplot(x='Global South', y='Deaths', data=year_2013, color='black', alpha=0.6)
plt.title('Confirmed Deaths per Development Status (2013)')
plt.xlabel('Development Status')
plt.ylabel('No. of Deaths')
plt.show()


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














    



