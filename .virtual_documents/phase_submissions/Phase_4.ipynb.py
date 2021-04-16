incidence_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/malaria_incidence.csv"
deaths_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/malaria_deaths.csv"
cases_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/malaria_confirmed_cases.csv"
country_regions_url = "https://meta.wikimedia.org/wiki/List_of_countries_by_regional_classification"
population_data_url = "https://raw.githubusercontent.com/mcuadera/info2950_project/master/datasets/population_data.csv"

incidence = pd.read_csv(incidence_url) # downloaded: 03/11/2021, last updated: 2020-03-27
deaths = pd.read_csv(deaths_url) #downloaded: 03/11/2021, last updated: 2018-12-20
cases = pd.read_csv(cases_url) #downloaded: 03/11/2021, last updated: 2018-12-20
population_data = pd.read_csv(population_data_url) #downloaded 03/16/2021, last updated: 2021-02-17
gdp_data = 
country_regions = requests.get(country_regions_url)
country_regions_table = pd.read_html(country_regions.text)[0] #tables of country regions


