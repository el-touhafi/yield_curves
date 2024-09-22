import datetime

from countryinfo import CountryInfo
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from playwright.sync_api import sync_playwright
from multiprocessing import Pool, cpu_count, freeze_support
from io import StringIO
from scipy.stats import norm
import sys


class YieldCurveScraper:
    """
    This class is designed to facilitate the scraping and creation of a dataframe
    containing yield information categorized by maturities for a specified date, denoted as X.
    It serves as a crucial tool for extracting and organizing data relevant to yield analysis.

    It's important to note a fundamental distinction in the data utilized for
    interpolating yields to construct the continuous yield curve and the data used for modeling.

    Additional details regarding the datasets employed for interpolation and modeling purposes
    will be thoroughly elucidated in dedicated classes named 'Interpolation' and 'Modeling'.
    These classes will provide comprehensive insights into the methodologies and datasets utilized for these specific tasks.
    NB : I have used two websites for scraping BANK AL MAGHRIB website and Tradingview.
    """

    def __init__(self, county: str, start_date: str):
        self.county = county
        # country code
        self.country_code = None
        # date to upload data for %d-%m-%Y
        self.start_date = start_date
        # url chosen based on pays variable to define it run the method URL
        self.url = None
        # the data corresponding to the date chosen to get it run scrape
        self.yiel_curve_data = None
        # maturities gotten from the scrape
        self.country_maturities = None

    def __str__(self):
        return f"pays : {self.county} \nDate de début : {self.start_date} \n"

    def code_C(self):
        """
        This method serves the purpose of initially selecting the country code to validate
        the input and ensure that a legitimate country has been entered.
        Its primary objective is to determine the appropriate URL to utilize based on the specified country.
        """
        country = CountryInfo(self.county)
        if len(country.alt_spellings()) == 0:
            print("You didn't written the country name correctly")
        else:
            self.country_code = country.alt_spellings()[0]

    def compute_url(self):
        """
        This method is dedicated to constructing the appropriate URL based on the provided date and country.
        Its functionality lies in generating a URL that corresponds specifically to the combination of the input date and country.
        By dynamically assembling the URL, this method ensures that the scraping process retrieves data pertinent
        to the specified country and date, facilitating accurate and targeted data extraction for further analysis.
        """
        if self.country_code == "MA":
            start_date = self.start_date.split("-")
            year = [
                date_elem
                for date_elem in start_date
                if len(date_elem) == 4
            ][0]
            month_index = (
                start_date.index(year) + 1
                if start_date.index(year) == 0
                else start_date.index(year) - 1
            )
            month = start_date[month_index]
            day = start_date[len(start_date) - (month_index + start_date.index(year))]
            self.url = f"https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor?date={day}%2F{month}%2F{year}"
        elif self.country_code != None:
            self.url = "https://fr.tradingview.com/markets/bonds/prices-all"

    def scrape_data(self):
        """
        This method is responsible for scraping the available data from the designated website.
        In the event of any malfunction, it suggests investigating potential alterations
        in the HTML code or data structure of the website.

        Additionally, ensuring the installation of requisite packages for executing requests to obtain the website's HTML is essential.
        This method acts as a pivotal step in the data retrieval process, necessitating vigilance
        in detecting and addressing any issues that may arise during scraping.
        """
        if self.country_code == "MA":
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                headers = {"User-Agent": "Custom"}
                page_data = requests.api.get(
                    self.url, verify=False, timeout=10, headers=headers
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
                    "class": "dynamic_contents_ref_12"
                },
            )
            if html_table != None:
                tmp_df = pd.read_html(self.url, flavor="lxml")[0]
                tmp_df.dropna(inplace=True)
                tmp_df.rename(columns={"Taux moyen pondéré": self.country_code + "_TBA"}, inplace=True)
                tmp_df.iloc[:, 2] = (
                    tmp_df.iloc[:, 2]
                    .str.replace("\xa0%", "")
                    .str.replace(",", ".")
                    .astype("float")
                    / 100.0
                )
                tmp_df["maturity"] = [
                    (
                        datetime.datetime.strptime(date_elem[0],"%d/%m/%Y") - datetime.datetime.strptime(self.start_date,"%d-%m-%Y")
                    ).days
                    for date_elem in tmp_df.values
                ]
                tmp_df = tmp_df[['maturity',self.country_code + '_TBA']]
                self.yiel_curve_data = tmp_df.reset_index(drop=True)
        elif self.country_code != None:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    browser.close()
            except Exception as e:
                if "Executable doesn't exist" in str(e):
                    print("Chromium isn't installed.")
                    print(
                        "To install Chromium, run the following command in your terminal:"
                    )
                    print("pip install playwright")
                    print("python -m playwright install")
                    sys.exit(1)
                else:
                    raise RuntimeError(f"An error occurred: {e}")
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page_data = browser.new_page()
                page_data.goto("https://fr.tradingview.com/markets/bonds/prices-all")
                page_data.wait_for_timeout(500)
                page_data.click('text="Charger plus"')
                page_data.wait_for_timeout(100)
                button_selector = "button.button-SFwfC2e0"
                while page_data.query_selector(button_selector) is not None:
                    yiel_curve_df = pd.read_html(
                        StringIO(
                            str(
                                bs(str(page_data.content()), "lxml").find(
                                    "div", {"class": "tableWrap-SfGgNYTG"}
                                )
                            )
                        ),
                        flavor="lxml",
                    )[0]
                    if yiel_curve_df.empty:
                        break
                    if any(yiel_curve_df.iloc[:, 0].str.contains(self.country_code)):
                        page_data.click(button_selector)
                        page_data.wait_for_timeout(300)
                        yiel_curve_df = pd.read_html(
                            StringIO(
                                str(
                                    bs(str(page_data.content()), "lxml").find(
                                        "div", {"class": "tableWrap-SfGgNYTG"}
                                    )
                                )
                            ),
                            flavor="lxml",
                        )[0]
                        break
                    page_data.click(button_selector)
                    page_data.wait_for_timeout(300)
                yiel_curve_df = pd.read_html(
                    StringIO(
                        str(
                            bs(str(page_data.content()), "lxml").find(
                                "div", {"class": "tableWrap-SfGgNYTG"}
                            )
                        )
                    ),
                    flavor="lxml",
                )[0]
                browser.close()
                if yiel_curve_df.empty:
                    print("This date doesn't contain any data")
                else:
                    yiel_curve_df.dropna(inplace=True)
                    yiel_curve_df = yiel_curve_df[
                        yiel_curve_df.iloc[:, 0].str.contains(self.country_code)
                    ]
                    self.country_maturities = (
                        yiel_curve_df.iloc[:, 0]
                        .str.split("Y", n=1, expand=True)
                        .iloc[:, 0]
                        + "Y"
                    )
                    self.country_maturities = (
                        self.country_maturities.to_frame().reset_index(drop=True)
                    )
                    yiel_curve_df = yiel_curve_df.iloc[:, [2, 4]]
                    yiel_curve_df.iloc[:, 0] = (
                        yiel_curve_df.iloc[:, 0]
                        .str.replace("%", "")
                        .str.replace("−", "-")
                        .str.replace(",", ".")
                        .astype("float")
                        / 100.0
                    )
                    yiel_curve_df.columns = [self.country_code + "_TBA", "maturity"]
                    yiel_curve_df.reset_index(drop=True, inplace=True)
                    if all(yiel_curve_df.iloc[:, 1].str.isdigit()):
                        yiel_curve_df.iloc[:, 1] = yiel_curve_df.iloc[:, 1].astype(
                            "int"
                        )
                    else:
                        all_maturities = yiel_curve_df.iloc[:, 1].str.split(" ").str
                        more_than_year_index = all_maturities[2].dropna().index[0]
                        less_than_year = list(
                            all_maturities[0][:more_than_year_index].astype("int")
                        )
                        more_than_year = list(
                            all_maturities[0][more_than_year_index:].astype("int") * 365
                            + all_maturities[2][more_than_year_index:].astype("int")
                        )
                        all_maturities = less_than_year
                        all_maturities.extend(more_than_year)
                        yiel_curve_df.iloc[:, 1] = all_maturities
                    self.yiel_curve_data = yiel_curve_df

    def compute(self):
        """
        this method will serve in computing all the methods needed to run the process and give the data needed
        """
        self.code_C()
        self.compute_url()
        self.scrape_data()

# p = YieldCurveScraper('USA','14-06-2024')
# p.compute()
# print(p.yiel_curve_data)