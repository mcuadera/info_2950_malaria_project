import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
plt.rcParams["figure.figsize"] = (10, 5)


incidence_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/master/datasets/malaria_incidence.csv"
deaths_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/master/datasets/malaria_deaths.csv"
cases_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/master/datasets/malaria_confirmed_cases.csv"
country_regions_url = "https://meta.wikimedia.org/wiki/List_of_countries_by_regional_classification"
population_data_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/master/datasets/population_data.csv"
gdp_data_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/main/datasets/gdppcppp_per_country.csv"
temp_data_url = "https://raw.githubusercontent.com/mcuadera/info_2950_malaria_project/main/datasets/temp_by_country.csv"

incidence = pd.read_csv(incidence_url) # downloaded: 03/11/2021, last updated: 2020-03-27
deaths = pd.read_csv(deaths_url) #downloaded: 03/11/2021, last updated: 2018-12-20
cases = pd.read_csv(cases_url) #downloaded: 03/11/2021, last updated: 2018-12-20
population_data = pd.read_csv(population_data_url) #downloaded 03/16/2021, last updated: 2021-02-17
gdp_data = pd.read_csv(gdp_data_url) #downloaded 04/16/2021, last updated: 2021-03-19
temp_data = pd.read_csv(temp_data_url) #downloaded 04/16/2021, last updated: 2022-12-24
country_regions = requests.get(country_regions_url)
country_regions_table = pd.read_html(country_regions.text)[0] #tables of country regions


population_data = population_data.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1).copy()
population_data = population_data.rename(columns={"Country Name":"Country"}).copy()


incidence_long = pd.melt(incidence, id_vars=["Country"], var_name="Year", value_name="Incidence")
incidence_long["Year"] = pd.to_datetime(incidence_long["Year"], format="get_ipython().run_line_magic("Y")", "")

deaths_long = pd.melt(deaths, id_vars=["Country"], var_name="Year", value_name="Deaths")
deaths_long["Year"] = pd.to_datetime(deaths_long["Year"], format="get_ipython().run_line_magic("Y")", "")

cases_long = pd.melt(cases, id_vars=["Country"], var_name="Year", value_name="Confirmed Cases")
cases_long["Year"] = pd.to_datetime(cases_long["Year"], format="get_ipython().run_line_magic("Y")", "")

gdp_long = pd.melt(gdp_data, id_vars=["Country"], var_name="Year", value_name="GDPpcPPP")
gdp_long["Year"] = pd.to_datetime(gdp_long["Year"], format="get_ipython().run_line_magic("Y")", "")

population_data_long = pd.melt(population_data, id_vars=["Country"], var_name="Year", value_name="Total Population")
population_data_long["Year"] = pd.to_datetime(population_data_long["Year"], format="get_ipython().run_line_magic("Y")", "")
population_data_long = population_data_long[(population_data_long["Year"] >= "2000-01-01")].copy()

temp_data["Date"] = pd.to_datetime(temp_data["Date"])
temp_data["Date"] = temp_data[temp_data["Date"] >= "2000-01-01"].copy()


temp_data_subset = temp_data[temp_data["Country"].isin(malaria_stat_merged["Country"].unique())] #only include WHO countries
temp_data_subset = temp_data_subset.set_index("Date")


temp_data_subset = temp_data_subset.groupby([temp_data_subset.index.year, "Country"])["AverageTemperature"].mean().reset_index() #annual average temp
temp_data_subset["Year"] = pd.to_datetime(temp_data_subset["Date"], format="get_ipython().run_line_magic("Y")", "")
temp_data_subset = temp_data_subset.drop("Date", axis=1).copy()
temp_data_subset = temp_data_subset.set_index("Year")


malaria_stat_merged = incidence_long.merge(deaths_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(cases_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(population_data_long, on=["Country","Year"])
malaria_stat_merged = malaria_stat_merged.merge(country_regions_table, on="Country")
malaria_stat_merged = malaria_stat_merged.merge(gdp_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(temp_data_subset, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.set_index("Year").copy()


malaria_stat_merged.head()


#Generating dummy variables for regions
subset_2013 = malaria_stat_merged.loc["2013-01-01"].copy()
subset_2013["Is Asia & Pacific"] = pd.get_dummies(subset_2013["Region"])["Asia & Pacific"]
subset_2013["Is Arab States"] = pd.get_dummies(subset_2013["Region"])["Arab States"]
subset_2013["Is Africa"] = pd.get_dummies(subset_2013["Region"])["Africa"]
subset_2013["Is South/Latin America"] = pd.get_dummies(subset_2013["Region"])["South/Latin America"]
subset_2013["Is Europe"] = pd.get_dummies(subset_2013["Region"])["Europe"]
subset_2013["Is Middle east"] = pd.get_dummies(subset_2013["Region"])["Middle east"]
subset_2013.head()


incidence_model_vars = ["Is Asia & Pacific", "Is Arab States", "Is Africa",
                        "Is South/Latin America", "Is Europe", "Is Middle east",
                        "AverageTemperature", "GDPpcPPP"]
subset_2013 = subset_2013.dropna(subset = incidence_model_vars).copy() # making sure there are no NA values

incidence_model_2013 = LinearRegression()
incidence_model_2013.fit(subset_2013[incidence_model_vars], subset_2013["Incidence"])
incidence_model_2013_coeff = incidence_model_2013.coef_[:]


for i in range(len(incidence_model_2013_coeff)):
    print('For', incidence_model_vars[i], 'variable, the regression coefficient is: {:.2f}'.format(incidence_model_2013_coeff[i]))


malaria_stat_merged_dropna = malaria_stat_merged.copy()
malaria_stat_merged_dropna["Is Asia & Pacific"] = pd.get_dummies(malaria_stat_merged_dropna["Region"])["Asia & Pacific"]
malaria_stat_merged_dropna["Is Arab States"] = pd.get_dummies(malaria_stat_merged_dropna["Region"])["Arab States"]
malaria_stat_merged_dropna["Is Africa"] = pd.get_dummies(malaria_stat_merged_dropna["Region"])["Africa"]
malaria_stat_merged_dropna["Is South/Latin America"] = pd.get_dummies(malaria_stat_merged_dropna["Region"])["South/Latin America"]
malaria_stat_merged_dropna["Is Europe"] = pd.get_dummies(malaria_stat_merged_dropna["Region"])["Europe"]
malaria_stat_merged_dropna["Is Middle east"] = pd.get_dummies(malaria_stat_merged_dropna["Region"])["Middle east"]
malaria_stat_merged_dropna = malaria_stat_merged_dropna.dropna(subset=incidence_model_vars).copy() # making sure there are no NA values

incidence_model_pooled = LinearRegression()
incidence_model_pooled.fit(malaria_stat_merged_dropna[incidence_model_vars], malaria_stat_merged_dropna["Incidence"])
incidence_model_pooled_coeff = incidence_model_pooled.coef_[:]


for i in range(len(incidence_model_pooled_coeff)):
    print('For', incidence_model_vars[i], 'variable, the regression coefficient is: {:.2f}'.format(incidence_model_pooled_coeff[i]))
