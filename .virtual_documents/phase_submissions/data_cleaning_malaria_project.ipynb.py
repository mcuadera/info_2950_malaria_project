import numpy as np
import pandas as pd
import datetime
import requests


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


malaria_stat_merged = incidence_long.merge(deaths_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(cases_long, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.merge(population_data_long, on=["Country","Year"])
malaria_stat_merged = malaria_stat_merged.merge(country_regions_table, on="Country")
malaria_stat_merged = malaria_stat_merged.merge(gdp_long, on=["Country", "Year"])


temp_data_subset = temp_data[temp_data["Country"].isin(malaria_stat_merged["Country"].unique())] #only include WHO countries
temp_data_subset = temp_data_subset.set_index("Date")
temp_data_subset = temp_data_subset.groupby([temp_data_subset.index.year, "Country"])["AverageTemperature"].mean().reset_index() #annual average temp
temp_data_subset["Year"] = pd.to_datetime(temp_data_subset["Date"], format="get_ipython().run_line_magic("Y")", "")
temp_data_subset = temp_data_subset.drop("Date", axis=1).copy()
temp_data_subset = temp_data_subset.set_index("Year")


malaria_stat_merged = malaria_stat_merged.merge(temp_data_subset, on=["Country", "Year"])
malaria_stat_merged = malaria_stat_merged.set_index("Year").copy()


malaria_stat_merged.head()


malaria_stat_merged.to_csv("../datasets/malaria_project_curated_data.csv")
