import scipy.linalg
import numpy as np


class CSP:
    def __init__(self, m_filters):
        self.m_filters = m_filters

    def fit(self, x_train, y_train):
        """
        Compute the CSP eigenvalues and CSP eigenvectors.

        :param x_train: train data, (n_trials, n_channels, n_samples)
        :param y_train: train labels for each trial, (n_trials)
        :returns eig_values: CSP eigenvalues with descending order, (n_channels)
        :returns u_mat: CSP eigenvectors with descending order, each row is a eigenvector, (n_channels, n_channels)
        """
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float)  # the parameter is '2' because the CSP method is designed for two classes
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))  # compute the covariance matrix (n_channels x n_channels) for the trial
            cov_x_trial /= np.trace(cov_x_trial)  # divide each element of cov_x_trial by the trace of 'cov_x_trial' to Standardization the covariance matrix
            cov_x[y_trial, :, :] += cov_x_trial  # compute the total covariance for two classes

        cov_x = np.asarray([cov_x[cls]/np.sum(y_labels == cls) for cls in range(2)])  # compute the mean covariance matrix for two classes
        cov_combined = cov_x[0]+cov_x[1]  # add the covariance matrix of two classes to build a covariance matrix of all data, which is (n_channels x n_channels)
        eig_values, u_mat = scipy.linalg.eig(cov_combined, cov_x[0])  # computed the eigenvalues and eigenvectors of the covariance matrix 'cov_combined' with respect to cov_x[0] (the first class)
        sort_indices = np.argsort(abs(eig_values))[::-1]  # get the index array 'sort_indices' sorted in descending order according to the abs of the eigenvalues
        eig_values = eig_values[sort_indices]  # reorder the eigenvalue array 'eig_values' in the order of the 'sort_indices'
        u_mat = u_mat[:, sort_indices]
        u_mat = np.transpose(u_mat)  # transpose to make each row of 'u_mat' is a CSP eigenvector

        return eig_values, u_mat

    def transform(self, x_trial, eig_vectors):
        """
        Select the first and last 'm_filters' channels (total is 2*m_filters), compute the CSP feature for each selected channel.

        :param x_trial: a trial of EEG data, (n_channels, n_samples)
        :param eig_vectors: the CSP eigenvectors with respect to one specific class in one specific frequency sub-band, (n_channels, n_channels)
        :returns np.log(var_z/sum_var_z): the CSP feature for each selected channel, (2*m_filters)
        """
        z_trial = np.matmul(eig_vectors, x_trial)  # 'z_trial' is (n_channels, n_samples)
        z_trial_selected = z_trial[:self.m_filters, :]  # select the first 'm_filters' channels
        z_trial_selected = np.append(z_trial_selected, z_trial[-self.m_filters:, :], axis=0)  # append the last 'm_filters' channels
        sum_z2 = np.sum(z_trial_selected**2, axis=1)  # compute the sum of z_trial^2 for each selected channel
        sum_z = np.sum(z_trial_selected, axis=1)  # compute the sum of z_trial for each selected channel
        var_z = (sum_z2 - (sum_z ** 2)/z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)  # compute the variance of each selected channel, the 'var_z' is (2*m_filters)
        sum_var_z = sum(var_z)
        return np.log(var_z/sum_var_z)  # the CSP feature for each selected channel
