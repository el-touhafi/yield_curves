import requests
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
from YieldCurveScraper import YieldCurveScraper
from tvDatafeed import TvDatafeed, Interval
from multiprocessing import Pool, cpu_count, freeze_support
from datetime import datetime
from bs4 import BeautifulSoup as bs


class ModelingDataScraper(YieldCurveScraper):
    """
    This class is designed to facilitate the scraping of data and creation
    of a dataframe containing the 1-month yield for a range of dates,
    spanning from a start to an end date. It plays a crucial role in gathering essential data
    for modeling the 1 month yield for a specefic country.

    However, it's important to note that the data used for this purpose differs significantly from the data
    utilized in the construction of the yield curve or the term structure curve.
    Further elaboration on the specifics of data for interpolation and the data modeling will be provided
    in separate classes dedicated to these tasks.

    NB: there's a specific tool or library for scrapping other datas for other countries than Morocco.
    you should download it and then the README and you will be capable of using it.
    git-url for the tool : https://github.com/rongardF/tvdatafeed
    """

    def __init__(self, country: str, start_date: str, end_date: str):
        YieldCurveScraper.__init__(self, country, start_date)
        # date to upload data until %d-%m-%Y
        self.end_date = end_date
        # a list of urls to extract the least maturities corresponding to computed dates
        self.modeling_data_urls = None
        # computed date list
        self.date_list = None
        self.model_data = None

    def compute_url(self):
        """
        This method is dedicated to constructing the appropriate URL based on the provided dates and country.
        By dynamically assembling the URLs, this method ensures that the scraping process retrieves data pertinent
        to the specified country and dates, facilitating accurate and targeted data extraction for further analysis.
        """
        if self.country_code == "MA":
            start_date = datetime.strptime(self.start_date, "%d-%m-%Y")
            end_date = datetime.strptime(self.end_date, "%d-%m-%Y")
            date_list = pd.date_range(start_date, end_date, freq="D")
            tmp_dataframe = pd.DataFrame(date_list, columns=["Date"])
            date_list = date_list.strftime("%d-%m-%Y")
            url = list()
            for i in date_list:
                year = [date_elem for date_elem in i.split("-") if len(date_elem) == 4][
                    0
                ]
                month_index = (
                    i.split("-").index(year) + 1
                    if i.split("-").index(year) == 0
                    else i.split("-").index(year) - 1
                )
                month = i.split("-")[month_index]
                url.append(
                    f"https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor/Marche-des-adjudications-des-bons-du-tresor/Taux-moyens-ponderes-mensuels-des-emissions-du-tresor?startMonth={month}&startYear={year}&endMonth={month}&endYear={year}"
                )
            self.modeling_data_urls = url
            self.date_list = tmp_dataframe

    def scrape_data(self, i_th_bkm_url=None):
        """
        This method is responsible for scraping the available data from the designated website.
        In the event of any malfunction, it suggests investigating potential alterations
        in the HTML code or data structure of the website.

        Additionally, ensuring the installation of requisite packages for executing requests to obtain the website's HTML is essential.
        This method acts as a pivotal step in the data retrieval process, necessitating vigilance
        in detecting and addressing any issues that may arise during scraping.
        """
        if self.country_code == "MA":
            try:
                headers = {"User-Agent": "Custom"}
                page_data = requests.api.get(
                    i_th_bkm_url, verify=False, timeout=10, headers=headers
                )
            except page_data.status_code as err:
                if err != 200:
                    print("Successful HTTP requests")
                else:
                    print(f"An HTTP error occurred: {err}")
            html_data = bs(page_data.text, "lxml")
            html_table = html_data.find(
                "table",
                {
                    "class": "weighted_average_rates_treasury_issues-standards dynamic_contents_ref_11"
                },
            )
            if html_table != None:
                tmp_df = pd.read_html(self.url, flavor="lxml")[0]
                tmp_df.dropna(inplace=True)
                tmp_df = tmp_df.T.iloc[1:]
                tmp_df.rename(columns={0: self.country_code + "_TBA"}, inplace=True)
                tmp_df.iloc[:, 0] = (
                    tmp_df.iloc[:, 0]
                    .str.replace("\xa0%", "")
                    .str.replace(",", ".")
                    .astype("float")
                    / 100.0
                )
                tmp_df["maturity"] = [
                    (
                        int(date_elem[0])
                        if ("semaine" in date_elem[1] or "week" in date_elem[1])
                        else int(date_elem[0]) * 365
                    )
                    for date_elem in tmp_df.index.str.split(" ")
                ]
                model_data = tmp_df.reset_index(drop=True)
                self.model_data = pd.read_csv('C://Users\Lenovo\Documents\GitHub\Backend\MA_BD.csv')
                if all(self.model_data.iloc[:, -1] != model_data.iloc[0, -1]):
                    print("s")
                    model_data.to_csv(
                        self.country_code + "_BD.csv",
                        mode="a",
                        header=False,
                        index=False,
                    )
                else:
                    return
        elif self.country_code != None:
            super().scrape_data()
            tv = TvDatafeed()
            # symbol = tv.search_symbol('TVC:'+self.country_maturities.iloc[0, 0])[0]["symbol"]
            # exchange = tv.search_symbol(self.country_maturities.iloc[0, 0])[0][
            #     "exchange"
            # ]
            end = datetime.strptime(self.end_date, "%d-%m-%Y")
            start = datetime.strptime(self.start_date, "%d-%m-%Y")
            self.model_data = tv.get_hist(
                symbol=self.country_maturities.iloc[0, 0],
                exchange='TVC',
                interval=Interval.in_daily,
                n_bars=(end - start).days + 200,
            )
            print((end - start).days + 200)
            self.model_data = self.model_data[["symbol", "close"]]
            self.model_data["symbol"] = [self.country_maturities.iloc[0, 0][2:5]] * len(
                self.model_data["symbol"]
            )
            self.model_data.rename(columns={"symbol": "maturity"}, inplace=True)
            s = self.country_maturities.iloc[0, 0][:2] + "_TBA"
            self.model_data.rename(columns={"close": s}, inplace=True)
            self.model_data.rename_axis(None, inplace=True)
            self.model_data = self.model_data[
                (self.model_data.index >= start) & (self.model_data.index <= end)
            ]
            self.model_data.index = self.model_data.index.strftime("%Y-%m-%d")
            self.model_data.iloc[:, 1] = (
                self.model_data.iloc[:, 1].astype("float") / 100.0
            )

    def load_data(self):
        """
        Due to the time complexity of computing the scrape method that i couldn't find
        other solution than this by adding a csv file named MA_BD for moroccan yield data to ensure downloading only the new data.
        NB : You can use also this method for other countries than morocco if you want to improve your computing speed. :)
        """
        if self.country_code == "MA":
            freeze_support()
            self.model_data = pd.read_csv('C://Users\Lenovo\Documents\GitHub\Backend\MA_BD.csv')
            self.model_data["Date"] = pd.to_datetime(
                self.model_data["Date"], format="%Y-%m-%d"
            )
            self.model_data.drop_duplicates(inplace=True, ignore_index=True)
            self.model_data.sort_values(by="Date")
            url1 = (
                np.array(self.modeling_data_urls)[
                    self.date_list[(self.date_list < self.model_data["Date"].iloc[-1])]
                    .dropna()
                    .index.tolist()
                ].tolist()
                + np.array(self.modeling_data_urls)[
                    self.date_list[(self.date_list > self.model_data["Date"].iloc[0])]
                    .dropna()
                    .index.tolist()
                ].tolist()
            )
            num_processes = cpu_count()
            if __name__ == "__main__":
                with Pool(num_processes - 3) as pool:
                    pool.map(self.scrape_data, url1)
                    pool.close()
                    pool.terminate()
                    pool.join()
            self.model_data = pd.read_csv('C://Users\Lenovo\Documents\GitHub\Backend\MA_BD.csv')
            self.model_data["Date"] = pd.to_datetime(
                self.model_data["Date"], format="%Y-%m-%d"
            )
            self.model_data.drop_duplicates(inplace=True, ignore_index=True)
            self.model_data.set_index("Date", drop=True, inplace=True)
            end = datetime.strptime(self.end_date, "%d-%m-%Y")
            start = datetime.strptime(self.start_date, "%d-%m-%Y")
            self.model_data.index = pd.to_datetime(
                self.model_data.index, format="%d-%m-%Y"
            )
            self.model_data = self.model_data[
                (self.model_data.index >= start) & (self.model_data.index <= end)
            ]
            self.model_data.sort_index(inplace=True)
            self.model_data = self.model_data.iloc[:, [1, 0]]

    def compute(self):
        """
        this method will serve in computing all the methods needed to run the process and give the data needed
        """
        super().code_C()
        self.compute_url()
        if self.country_code == "MA":
            self.load_data()
        else:
            self.scrape_data()



# a = ModelingDataScraper('US','13-5-2024','20-5-2024')
# a.compute()
# print(a.model_data)