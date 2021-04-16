import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
from scipy import stats
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 5)


incidence_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/malaria_incidence.csv"
deaths_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/malaria_deaths.csv"
cases_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/malaria_confirmed_cases.csv"
country_regions_url = "https://meta.wikimedia.org/wiki/List_of_countries_by_regional_classification"
population_data_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/population_data.csv"

incidence = pd.read_csv(incidence_url) # downloaded: 03/11/2021, last updated: 2020-03-27
deaths = pd.read_csv(deaths_url) #downloaded: 03/11/2021, last updated: 2018-12-20
cases = pd.read_csv(cases_url) #downloaded: 03/11/2021, last updated: 2018-12-20
population_data = pd.read_csv(population_data_url) #downloaded 03/16/2021, last updated: 2021-02-17

country_regions = requests.get(country_regions_url)
country_regions_table = pd.read_html(country_regions.text)[0] #tables of country regions


population_data = population_data.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1).copy()
population_data = population_data.rename(columns={"Country Name":"Country"}).copy()


incidence_long = pd.melt(incidence, id_vars=["Country"], var_name="Year", value_name="Incidence")
incidence_long["Year"] = incidence_long["Year"].astype("int")

deaths_long = pd.melt(deaths, id_vars=["Country"], var_name="Year", value_name="Deaths")
deaths_long["Year"] = deaths_long["Year"].astype("int")

cases_long = pd.melt(cases, id_vars=["Country"], var_name="Year", value_name="Confirmed Cases")
cases_long["Year"] = cases_long["Year"].astype("int")

population_data_long = pd.melt(population_data, id_vars=["Country"], var_name="Year", value_name="Total Population")
population_data_long["Year"] = population_data_long["Year"].astype("int")
population_data_long = population_data_long[population_data_long["Year"].between(2000, 2017)].copy()


malaria_stat_merged = incidence_long.merge(deaths_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(cases_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(population_data_long, on=["Country","Year"])
malaria_stat_merged = malaria_stat_merged.merge(country_regions_table, on="Country")


malaria_stat_merged.head()


by_year = malaria_stat_merged.groupby("Year")


plt.plot(by_year["Deaths"].mean(), label="Mean")
plt.plot(by_year["Deaths"].median(), label="Median")
plt.title("Deaths per Year")
plt.xlabel("Year")
plt.xticks(rotation = 40)
plt.ylabel("Deaths")
plt.legend()
plt.show()


plt.plot(by_year["Deaths"].median(), color="orange")
plt.title("Median Deaths per Year")
plt.xlabel("Year")
plt.xticks(rotation = 40)
plt.ylabel("Median Deaths")
plt.show()


by_region = malaria_stat_merged.groupby(["Region","Year"])


by_region.mean()["Deaths"].unstack().plot(kind="bar")
plt.title("Average Deaths per Region (2000-2017)")
plt.xlabel("Region")
plt.xticks(rotation=40)
plt.ylabel("Average Deaths")
plt.legend(loc='upper center', bbox_to_anchor=(1.5, 0.8), shadow=True, ncol=4)
plt.show()


by_region.median()["Deaths"].unstack().plot(kind="bar")
plt.title("Median Deaths per Region (2000-2017)")
plt.xlabel("Region")
plt.xticks()
plt.ylabel("Median Deaths")
plt.legend(loc='upper center', bbox_to_anchor=(1.5, 0.8), shadow=True, ncol=4)
plt.show()


africa = malaria_stat_merged[malaria_stat_merged["Region"]=="Africa"]


africa_2017 = africa[africa["Year"]==2017]
plt.bar(africa_2017["Country"], africa_2017["Deaths"])
plt.title("Total Deaths per African Country 2017")
plt.xlabel("Country")
plt.xticks(rotation=90)
plt.ylabel("Total Deaths")
plt.show()


africa_2000 = africa[africa["Year"]==2000]
plt.bar(africa_2000["Country"], africa_2000["Deaths"])
plt.title("Total Deaths per African Country 2000")
plt.xlabel("Country")
plt.xticks(rotation=90)
plt.ylabel("Total Deaths")
plt.show()


angola = africa[africa["Country"]=="Angola"]
kenya = africa[africa["Country"]=="Kenya"]
plt.plot(angola["Year"], angola["Deaths"], label="Angola")
plt.plot(kenya["Year"], kenya["Deaths"], label="Kenya")
plt.title("Deaths per Year")
plt.xlabel("Year")
plt.xticks(rotation = 40)
plt.ylabel("Deaths")
plt.legend()
plt.show()


year_region = malaria_stat_merged.groupby(["Year","Region"])
year_region["Incidence"].mean().unstack().plot()
plt.title("Average Incidence Per Year")
plt.xlabel("Year")
plt.ylabel("Incidence (cases per 1000 people)")
plt.show()


year_region = malaria_stat_merged.groupby(["Year","Region"])
year_region["Incidence"].median().unstack().plot()
plt.title("Median Incidence Per Year")
plt.xlabel("Year")
plt.ylabel("Incidence (cases per 1000 people)")
plt.show()


malaria_corr_matrix = malaria_stat_merged.corr()
sns.heatmap(malaria_corr_matrix, cmap="YlGnBu", annot=True)
plt.show()


plt.scatter(malaria_stat_merged["Confirmed Cases"], malaria_stat_merged['Deaths'])
plt.title("Malaria Deaths vs Confirmed Cases of Malaria")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.show()
