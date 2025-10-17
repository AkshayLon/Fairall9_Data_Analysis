import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

class FairallDataManager:

    def __init__(self, file_name: str):
        self.raw_data = pd.read_csv(file_name)

    def subset_by_telescope(self, telescope_name: str) -> pd.DataFrame:
        """
        Possible telescope name inputs:
        UVOT - Swift Ultraviolet/Optical Telescope
        XRT - X-ray Telescope
        """
        return self.raw_data[self.raw_data['Telescope'] == telescope_name]
    
    def plot_light_curve(self, telescope: str, band: str):
        """
        Plots the light curve for a given telescope and band.
        """
        subset = self.subset_by_telescope(telescope)
        band_data = subset[subset['Band'] == band]
        
        plt.figure(figsize=(10, 5))
        plt.errorbar(band_data['MJD'], band_data['Flux'], yerr=band_data['Error'], fmt='o', label=f'{telescope} - {band}')
        plt.xlabel('MJD')
        plt.ylabel('Flux')
        plt.title(f'Light Curve for {telescope} in {band} Band')
        plt.legend()
        plt.show()

    def interpolate_light_curve(self, telescope: str, band: str, leave_out_data: bool = False):
        """
        Performs Gaussian Process Regression on the Fairall light curve data for a given telescope and band.
        Fills in the missing data by interpolation.
        """
        subset = self.subset_by_telescope(telescope)
        band_data = subset[subset['Band'] == band]

        # Prepare the data for GP Regression
        X = band_data['MJD'].values.reshape(-1, 1)
        y = band_data['Flux'].values

        """
        In this example, we use a Rational Quadratic kernel as per research.
        Mental note to make this configurable as per project progress discussion.
        """
        rq_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        gp = GaussianProcessRegressor(kernel=rq_kernel)

        # Interpolate the data
        gp.fit(X, y)

        # Get predictions along with std_dev statistics
        X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred, sigma = gp.predict(X_pred, return_std=True)

        # Plot results
        plt.figure(figsize=(10, 5))
        if not leave_out_data:
            plt.errorbar(band_data['MJD'], band_data['Flux'], yerr=band_data['Error'], fmt='o', label='Observed Data')
        plt.plot(X_pred, y_pred, 'r-', label='GP mean prediction')
        plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='r', label='95 percent band')
        plt.xlabel('MJD')
        plt.ylabel('Flux')
        plt.title(f'Gaussian Process Interpolation for {telescope} in {band} Band')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    fairall_data = FairallDataManager("F9LCs.csv")
    fairall_data.plot_light_curve("UVOT", "V")