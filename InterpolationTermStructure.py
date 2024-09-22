import sys

sys.path.append("..")
from scipy.optimize import minimize
from YieldCurveScraper import YieldCurveScraper
import numpy as np
import pandas as pd


class InterpolationTermStructure(YieldCurveScraper):
    """
    This class is designed to interpolate missing values for
    the daily data of interest across different maturities,
    ultimately constructing the term structure of the interest rate.
    NB: that understanding the interpolation process requires familiarity with
    linear and cubic functions, as well as Nelson-Siegel and Nelson-Siegel-Svensson functions.
    Additionally, you should be knowledgeable about parameter optimization through error minimization,
    employing numerical derivation and numerical optimization techniques.
    """

    def __init__(self, country: str, start_date: str, method: str):
        super().__init__(country, start_date)
        self._linear_interpolation = pd.DataFrame()
        self._cubic_interpolation = pd.DataFrame()
        self._nelson_siegel = pd.DataFrame()
        self._nelson_siegel_svensson = pd.DataFrame()
        self._linear_frames = pd.DataFrame()
        self.method = method

    def compute(self):
        """
        This method match case for the interpolation needed
        """
        super().compute()
        match self.method:
            case "bootsrapping":
                return self.linear_frames
            case "linear_interpolation":
                return self.linear_interpolation
            case "cubic_interpolation":
                return self.cubic_interpolation
            case "nelson_siegel":
                return self.nelson_siegel
            case "nelson_siegel_svensson":
                return self.nelson_siegel_svensson
            case _:
                print(
                    'method in ["bootsrapping","linear_interpolation","cubic_interpolation","nelson_siegel","nelson_siegel_svensson"]'
                )

    @property
    def linear_frames(self):
        """
        The method applies linear interpolation between each point of the term structure of interest rate.
        Equation : a * maturity + b = f(maturity) = interest rate  (for each [interest rate(i),interest rate(i+1)])
        """
        data_copy = self.yiel_curve_data.copy()
        for j in range(len(data_copy.maturity)):
            if j < len(data_copy.maturity) - 1:
                bootstrap = []
                for i in np.arange(
                    data_copy.iloc[j, 0] + 1, data_copy.iloc[j + 1, 0], 5
                ):
                    bootstrap.append(
                        (
                            (data_copy.iloc[j + 1, 0] - i) * data_copy.iloc[j, -1]
                            + (i - data_copy.iloc[j, 0]) * data_copy.iloc[j + 1, -1]
                        )
                        / (data_copy.iloc[j + 1, 0] - data_copy.iloc[j, 0])
                    )
                bootstrap_df = pd.DataFrame(
                    np.arange(data_copy.iloc[j, 0] + 1, data_copy.iloc[j + 1, 0], 5),
                    columns=["maturity"],
                )
                bootstrap_df[self.country_code + "_TBA"] = bootstrap
                bootstrap_df = bootstrap_df.iloc[:, [0, -1]]
                data_copy = pd.concat([data_copy, bootstrap_df])
        #
        self._linear_frames = data_copy.drop_duplicates().sort_values(
            by="maturity", ignore_index=True
        )
        #
        return self._linear_frames

    @property
    def linear_interpolation(self):
        """
        The method applies linear regression to the term structure of interest rates using least squares minimization.
        Equation : a * maturity + b = f(maturity) = interest rate
        """
        data_copy = self.yiel_curve_data.copy()
        maturity_years = data_copy["maturity"] / 360
        linear_matrix = np.array([np.ones(len(maturity_years)), maturity_years]).astype(
            float
        )
        linear_matrix_t = linear_matrix.transpose()
        slope = np.linalg.inv(np.dot(linear_matrix, linear_matrix_t))
        slope = np.dot(np.dot(slope, linear_matrix), np.array(data_copy[self.country_code + '_TBA']))
        linear_interpolation_df = pd.DataFrame(
            np.arange(data_copy.iloc[0]['maturity'] + 1, data_copy.iloc[-1]['maturity'], 5),
            columns=["maturity"],
        )
        maturity_years = linear_interpolation_df["maturity"] / 360
        linear_matrix = np.array([np.ones(len(maturity_years)), maturity_years])
        linear_interpolation_df[self.country_code + "_TBA"] = np.dot(
            slope, linear_matrix
        )
        linear_interpolation_df = linear_interpolation_df.iloc[:, [1, 0]]
        #
        self._linear_interpolation = linear_interpolation_df.sort_values(
            by="maturity", ignore_index=True
        )
        #
        return self._linear_interpolation

    @property
    def cubic_interpolation(self):
        """
        The method applies cubic regression to the term structure of interest rates using least squares minimization.
        Equation : a * maturity ** 3 + b * maturity ** 2 + c * maturity + d = f(maturity) = interest rate
        """
        data_copy = self.yiel_curve_data.copy()
        maturity_years = data_copy["maturity"] / 360
        cubic_matrix = np.array(
            [
                np.ones(len(maturity_years)),
                maturity_years,
                (maturity_years) ** 2,
                (maturity_years) ** 3,
            ]
        ).astype(float)
        cubic_matrix_t = cubic_matrix.transpose()
        params = np.linalg.inv(np.dot(cubic_matrix, cubic_matrix_t))
        params = np.dot(np.dot(params, cubic_matrix), np.array(data_copy[self.country_code + "_TBA"]))
        cubic_interpolation_df = pd.DataFrame(
            np.arange(data_copy.iloc[0, -1] + 1, data_copy.iloc[-1, -1], 5),
            columns=["maturity"],
        )
        maturity_years = cubic_interpolation_df["maturity"] / 360
        cubic_matrix = np.array(
            [
                np.ones(len(maturity_years)),
                maturity_years,
                (maturity_years) ** 2,
                (maturity_years) ** 3,
            ]
        )
        cubic_interpolation_df[self.country_code + "_TBA"] = np.dot(
            params, cubic_matrix
        )
        cubic_interpolation_df = cubic_interpolation_df.iloc[:, [1, 0]]
        #
        self._cubic_interpolation = cubic_interpolation_df.sort_values(
            by="maturity", ignore_index=True
        )
        #
        return self._cubic_interpolation

    def nelson_siegel_algo(self, ns_params):
        """
        The method applies Nelson Siegel to the point of interest rates
        to interpolate the term structure using least squares minimization.
        Equation :
        fact0 + fact1 (1 - exp(- maturity/tau)) / (t/tau) +
        fact2 * ((1 - exp(- maturity/tau)) / (t/tau) - exp(- maturity/tau))
        = f(maturity) = interest rate
        """
        data_copy = self.yiel_curve_data.copy()
        maturity_years = (data_copy["maturity"] / 360).astype(float)
        equation1 = (1 - np.exp(-(maturity_years * ns_params[-1]))) / (
            maturity_years * ns_params[-1]
        )
        ns_matrix = np.array(
            [
                np.ones(len(maturity_years)),
                equation1,
                (equation1 - np.exp(-maturity_years * ns_params[-1])),
            ]
        )
        ns_equation = np.dot(ns_params[0:-1], ns_matrix)
        error = sum((ns_equation - data_copy[self.country_code + "_TBA"]) ** 2)
        return error

    @property
    def nelson_siegel(self):
        """
        The BFGS method is used for approximating the parameters of nelson siegel formula
        check out the method it's called also quasi-Newton's Method wich consist on
        Hessian matrix.
        """
        optimal_params = minimize(
            self.nelson_siegel_algo, [0.01, 0.01, 0.01, 1], method="BFGS", tol=1e-10
        )
        data_copy = self.yiel_curve_data.copy()
        nelson_siegel_df = pd.DataFrame(
            np.arange(data_copy.iloc[0]['maturity'] + 1, data_copy.iloc[-1]['maturity'], 5),
            columns=["maturity"],
        )
        maturity_years = nelson_siegel_df["maturity"] / 360
        equation1 = (1 - np.exp(-optimal_params.x[-1] * maturity_years)) / (
            optimal_params.x[-1] * maturity_years
        )
        ns_matrix = np.array(
            [
                np.ones(len(maturity_years)),
                equation1,
                equation1 - np.exp(-optimal_params.x[-1] * maturity_years),
            ]
        )
        nelson_siegel_df[self.country_code + "_TBA"] = np.dot(
            optimal_params.x[0:-1], ns_matrix
        )
        nelson_siegel_df = nelson_siegel_df.iloc[:, [1, 0]]
        #
        self._nelson_siegel = nelson_siegel_df.sort_values(
            by="maturity", ignore_index=True
        )
        #
        return self._nelson_siegel

    def nelson_siegel_svensson_algo(self, params):
        """
        The method applies Nelson Siegel Svensson to the point of interest rates
        to interpolate the term structure using least squares minimization.
        Equation :
        fact0 + fact1 (1 - exp(- maturity/tau)) / (t/tau) +
        fact2 * ((1 - exp(- maturity/tau)) / (t/tau) - exp(- maturity/tau)) +
        fact2 * ((1 - exp(- maturity/lambda)) / (t/lambda) - exp(- maturity/lambda))
        = f(maturity) = interest rate
        """
        data_copy = self.yiel_curve_data.copy()
        maturity_years = (data_copy["maturity"] / 360).astype(float)
        equation1 = (1 - np.exp(-(maturity_years * params[-1]))) / (
            maturity_years * params[-1]
        )
        equation2 = (1 - np.exp(-(maturity_years * params[-2]))) / (
            maturity_years * params[-2]
        )
        nss_matrix = np.array(
            [
                np.ones(len(maturity_years)),
                equation2,
                (equation2 - np.exp(-maturity_years * params[-2])),
                (equation1 - np.exp(-maturity_years * params[-1])),
            ]
        )
        nss_equation = np.dot(params[0:-2], nss_matrix)
        error = sum((nss_equation - data_copy[self.country_code + "_TBA"]) ** 2)
        return error

    @property
    def nelson_siegel_svensson(self):
        """
        The BFGS method is used for approximating the parameters of nelson siegel formula
        check out the method it's called also quasi-Newton's Method wich consist on
        Hessian matrix.
        """
        otimal_params = minimize(
            self.nelson_siegel_svensson_algo,
            [0.1, 0.01, 0.01, 0.01, 1, 2],
            method="BFGS",
            tol=1e-10,
        )
        data_copy = self.yiel_curve_data.copy()
        nelson_siegel_svensson_df = pd.DataFrame(
            np.arange(data_copy.iloc[0]['maturity'] + 1, data_copy.iloc[-1]['maturity'], 5),
            columns=["maturity"],
        )
        maturity_years = nelson_siegel_svensson_df["maturity"] / 360
        equation1 = (1 - np.exp(-(maturity_years * otimal_params.x[-1]))) / (
            maturity_years * otimal_params.x[-1]
        )
        equation2 = (1 - np.exp(-(maturity_years * otimal_params.x[-2]))) / (
            maturity_years * otimal_params.x[-2]
        )
        nss_matrix = np.array(
            [
                np.ones(len(maturity_years)),
                equation2,
                (equation2 - np.exp(-maturity_years * otimal_params.x[-2])),
                (equation1 - np.exp(-maturity_years * otimal_params.x[-1])),
            ]
        )
        nelson_siegel_svensson_df[self.country_code + "_TBA"] = np.dot(
            otimal_params.x[0:-2], nss_matrix
        )
        nelson_siegel_svensson_df = nelson_siegel_svensson_df.iloc[:, [1, 0]]
        #
        self._nelson_siegel_svensson = nelson_siegel_svensson_df.sort_values(
            by="maturity", ignore_index=True
        )
        #
        return self._nelson_siegel_svensson

p = InterpolationTermStructure('US','23-08-2024','nelson_siegel_svensson')
p.compute()
print(p.nelson_siegel_svensson)
import matplotlib.pyplot as plt
plt.plot(p.nelson_siegel_svensson.iloc[:,1],p.nelson_siegel_svensson.iloc[:,0])
plt.show()

"""
bootsrapping","linear_interpolation","cubic_interpolation","nelson_siegel","nelson_siegel_svensson
"""