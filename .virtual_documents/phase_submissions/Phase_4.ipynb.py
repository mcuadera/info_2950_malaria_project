import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

plt.rcParams["figure.figsize"] = (10, 5)


curated_df_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/main/datasets/malaria_project_curated_data.csv"
malaria_df = pd.read_csv(curated_df_url, index_col=0, parse_dates=True)
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


year_2013.groupby('Region').mean().round(2)


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


malaria_df["Confirmed Cases"].plot(kind="hist")
plt.title("Confirmed Cases of Malaria")
plt.xlabel("Confirmed Cases")
plt.ylabel("Count")


sns.boxplot(x = year_2013["Region"], y = year_2013["Confirmed Cases"])
sns.stripplot(x=year_2013["Region"],y= year_2013["Confirmed Cases"], data=year_2013, color="black", alpha=0.4)
plt.xlabel("Region")
plt.ylabel("Confirmed Cases")
plt.title("Confirmed Cases per Region")
plt.show()


sns.boxplot(x="Region", y ="Confirmed Cases", data=year_2013[year_2013['Region']get_ipython().getoutput("='Africa'])")
sns.stripplot(x="Region", y ="Confirmed Cases", data=year_2013[year_2013['Region']get_ipython().getoutput("='Africa'], color="black", alpha=0.4)")
plt.xlabel("Region")
plt.ylabel("Confirmed Cases")
plt.title("Confirmed Cases per Region (excluding Africa)")
plt.show()


sns.boxplot(x=year_2013["Global South"], y=year_2013["Confirmed Cases"])
sns.stripplot(x=year_2013["Global South"], y=year_2013["Confirmed Cases"], data=year_2013, color="black", alpha=0.4)
plt.xlabel("Development Status")
plt.ylabel("Confirmed Cases")
plt.title("Confirmed Cases per Development Status")
plt.show()


sns.lineplot(x = malaria_df.index, y=malaria_df["AverageTemperature"], hue='Region', data=malaria_df, ci=None)
plt.title('Average Temperatures Over Time per Region (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average Temperature (Celsius)')
plt.legend()
plt.show()


sns.lineplot(x = malaria_df.index, y=malaria_df["AverageTemperature"], data=malaria_df)
plt.title('Average Temperatures Over Time(2000-2013)')
plt.xlabel('Year')
plt.ylabel('Average Temperature (Celsius)')
plt.show()


year_2013["GDPpcPPP"].plot(kind="hist")
plt.title("GDP Per Capita PPP")
plt.xlabel("GDP Per Capita PPP")
plt.ylabel("Count")
plt.show()


sns.boxplot(x = year_2013["Global South"], y = year_2013["GDPpcPPP"])
sns.stripplot(x=year_2013["Global South"],y= year_2013["GDPpcPPP"], data=year_2013, color="black", alpha=0.4)
plt.xlabel("Development Status")
plt.ylabel("GDP Per Capita PPP")
plt.title("GDP Per Capita PPP by Development Status")
plt.show()


sns.boxplot(x = year_2013["Region"], y = year_2013["GDPpcPPP"])
sns.stripplot(x=year_2013["Region"],y= year_2013["GDPpcPPP"], data=year_2013, color="black", alpha=0.4)
plt.xlabel("Region")
plt.ylabel("GDP Per Capita PPP")
plt.title("GDP Per Capita PPP by Region")
plt.show()


sns.scatterplot(x="GDPpcPPP", y="Incidence", hue="Region", data=malaria_df)
plt.xlabel("GDP Per Capita PPP")
plt.ylabel("Incidence")
plt.title("GDP Per Capita PPP vs Malaria Incidence")
plt.show()


regions = pd.get_dummies(malaria_df['Region']) # generating dummy variables for regions
malaria_df = pd.concat([malaria_df, regions], axis=1).copy()

incidence_model_vars = ["Asia & Pacific", "Arab States", "Africa",
                        "South/Latin America", "Europe", "Middle east",
                        "AverageTemperature", "GDPpcPPP"]

malaria_df_no_na = malaria_df.dropna(subset = incidence_model_vars).copy() # making sure there are no NA values
malaria_df_no_na.head()
print('Without dropping any NAs, there are {} columns'.format(malaria_df.shape[0]))
print('After dropping NAs for each of the variables of interest,there are {} columns'.
      format(malaria_df_no_na.shape[0]))
print('Dropping NAs did not significantly impact the number of variables in our dataset')


for i in incidence_model_vars:
    linear_model = LinearRegression()
    linear_model.fit(malaria_df_no_na[[i]], malaria_df_no_na['Incidence'])
    b = linear_model.coef_[0]
    r2 = linear_model.score(malaria_df_no_na[[i]], malaria_df_no_na['Incidence'])
    
    print("The predictor variable is:", i)
    print("The slope of the model is: {:.2f}".format(b))
    print("The r^2 of the model is: {:.2f} \n".format(r2))


X = malaria_df_no_na[incidence_model_vars]
Y = malaria_df_no_na['Incidence']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)


incidence_training_model = LinearRegression()
incidence_training_model.fit(X_train, Y_train)
incidence_training_model_coeff = incidence_training_model.coef_[:]
print('The r^2 of the multivariate regression model using the training data is: {:.2f}'.
      format(incidence_training_model.score(X_train, Y_train)))


for i in range(len(incidence_training_model_coeff)):
    print('For', incidence_model_vars[i], 'variable, the regression coefficient is: {:.2f}'.format(incidence_training_model_coeff[i]))


incidence_prediction = incidence_training_model.predict(X_test)
mae = mean_absolute_error(Y_test, incidence_prediction)
print('The mean absolute error of our incidence model is: {:.2f}'.format(mae))


twoCols = malaria_df[["GDPpcPPP", "Incidence"]]
twoCols = twoCols.dropna(axis=0)
kmeans = KMeans(n_clusters=6)
y_kmeans = kmeans.fit(twoCols)

centers = kmeans.cluster_centers_
sns.scatterplot(x="GDPpcPPP", y="Incidence", hue="Region", data=malaria_df)
plt.scatter(centers[:, 0], centers[:, 1], c='black')

plt.xlabel("GDP pc PPP")
plt.ylabel("Malaria Incidence")
plt.show()


le_region = preprocessing.LabelEncoder()
malaria_df_no_na['Region Labelled'] = le_region.fit_transform(malaria_df_no_na['Region'])
malaria_df_no_na.head()


region_model_vars = ['GDPpcPPP', 'Incidence', 'AverageTemperature']
X = malaria_df_no_na[region_model_vars]
Y = malaria_df_no_na['Region Labelled']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)


region_training_model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=2200).fit(X_train, Y_train)
region_training_model_coeff = region_training_model.coef_[:]
print('The r^2 of the multivariate regression model using the training data is: {:.2f}'.
      format(region_training_model.score(X_train, Y_train)))


region_predict = region_training_model.predict(X_test)
accuracy = accuracy_score(Y_test, region_predict)
print('Accuracy of the model: {:.2f}get_ipython().run_line_magic("'.format(accuracy*100))", "")


incidence_region_model = ols('Incidence ~ Region', data=year_2013).fit()
aov_table = sm.stats.anova_lm(incidence_region_model)
print(aov_table)


incidence_region_model = ols('Incidence ~ Region', data=year_2013[year_2013['Region']get_ipython().getoutput("='Africa']).fit()")
aov_table = sm.stats.anova_lm(incidence_region_model)
print(aov_table)
