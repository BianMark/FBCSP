import numpy as np
from bin.CSP import CSP


class FBCSP:
    def __init__(self, m_filters):
        self.m_filters = m_filters
        self.fbcsp_filters_multi = []

    def fit(self, x_train_fb, y_train):
        """
        Get the dictionary 'fbcsp_filters_multi' which contains multiple groups of filter banks for all classes, each group contains the CSP eigenvalues and CSP eigenvectors with respect to one specific class in all frequency sub-bands.

        :param x_train_fb: train data decomposed to each frequency sub-bands, (n_fbanks, n_trials, n_channels, n_samples)
        :param y_train: train labels for each trial, (n_trials)
        """
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            """
            Get many filter banks, each filter bank contains the CSP eigenvalues and CSP eigenvectors with respect to 'y_train_cls' in a specific frequency sub-band.

            :param x_train_fb: train data decomposed to each frequency sub-bands, (n_fbanks, n_trials, n_channels, n_samples)
            :param y_train_cls: train labels for each trial, the label of all classes is '1' expect for one chosen class (label of the chosen class is '0'), (n_trials)
            :returns fbcsp_filters: a dictionary contains the CSP eigenvalues and CSP eigenvectors for the input data in each frequency sub-bands,
                'j': the index of each frequency sub-band,
                'eig_val': the CSP eigenvalues,
                'u_mat': the CSP eigenvectors
            """
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):  # for each frequency sub-band
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters = get_csp(x_train_fb, y_train_cls)  # a filter bank with respect to the class 'y_classes_unique[i]'
            self.fbcsp_filters_multi.append(fbcsp_filters)

    def transform(self, x_data, class_idx=0):
        """
        Get the CSP features for each trial in each frequency band.

        :param x_data: EEG data decomposed to each frequency sub-bands, (n_fbanks, n_trials, n_channels, n_samples)
        :param class_idx: the index of the chosen class
        :returns x_features: the CSP feature for each trial in each frequency band, each frequency band has 2*m_filters CSP features for one specific trial, (n_trials, n_fbanks*m_filters*2)
        """
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros((n_trials, self.m_filters*2*len(x_data)), dtype=np.float)  # len(x_data) = n_fbanks
        for i in range(n_fbanks):
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            # eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            for k in range(n_trials):
                x_trial = np.copy(x_data[i, k, :, :])
                csp_feat = self.csp.transform(x_trial, eig_vectors)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 2] = csp_feat[j]  # the first 'm_filters' CSP features are stored
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 1] = csp_feat[-j-1]  # the last 'm_filters' CSP features are stored

        return x_features
