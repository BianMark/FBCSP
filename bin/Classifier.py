import numpy as np


class NBPW:
    def fit(self, x_features, y_train):
        self.classes = np.sort(np.unique(y_train))
        self.n_classes = len(self.classes)
        self.n_features = x_features.shape[1]
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.std = np.zeros((self.n_classes, self.n_features))
        self.prior = np.zeros(self.n_classes)
        self.n = np.zeros(self.n_classes)
        self.x_bars = []
        for i, cl in enumerate(self.classes):
            x_features_cl = x_features[y_train == cl]  # (n_selected_trials, n_features)
            self.mean[i] = np.mean(x_features_cl, axis=0)  # (n_classes, n_features)
            self.std[i] = np.std(x_features_cl, axis=0)  # (n_classes, n_features)
            self.prior[i] = x_features_cl.shape[0] / x_features.shape[0]  # (n_classes)
            self.n[i] = len(x_features_cl)
            self.x_bars.append(x_features_cl)  # x_bars = (n_classes, n_selected_trials(could be different in each class), n_features)

    def predict(self, x_features):
        n_trials = x_features.shape[0]
        y_predicted = np.zeros(n_trials)
        for i in range(n_trials):
            prob = np.zeros(self.n_classes)
            for j, cl in enumerate(self.classes):
                p_X_ci = 1
                for d in range(self.n_features):
                    h = (4 / (3 * self.n[j])) ** (1 / 5) * self.std[j][d]
                    p_xi_ci = 0
                    for w in range(len(self.x_bars[j])):
                        p_xi_ci += (1 / (2 * np.pi) ** (0.5)) * np.exp(-0.5 * (((x_features[i, d] - self.x_bars[j][w][d]) / h) ** 2))
                    p_xi_ci = p_xi_ci / self.n[j]
                    p_X_ci *= p_xi_ci
                prob[j] = np.log(self.prior[j]) + np.log(p_X_ci)
            y_predicted[i] = self.classes[np.argmax(prob)]
        return y_predicted


class Classifier:
    def __init__(self, model, n_features_select=4, n_csp_pairs=2):
        self.model = model  # 'model' is the type of classifier, including 'SVR', 'SVM', 'rf', 'lr' etc
        self.feature_selection = False
        self.n_features_select = n_features_select
        self.n_csp_pairs = n_csp_pairs

    def predict(self, x_features):
        """
        Predict the class based on the features of train data.

        :param x_features: the CSP feature for each trial in each frequency band, each frequency band has 2*m_filters CSP features for one specific trial, (n_trials, n_fbanks*m_filters*2)
        :returns y_predicted: the predicted classes based on the features, (n_trials)
        """
        if self.feature_selection:
            x_features_selected = self.feature_selection.transform(x_features)
        else:
            x_features_selected = x_features
        y_predicted = self.model.predict(x_features_selected)
        return y_predicted

    def fit(self, x_features, y_train):
        """
        Train the classifier based on the features of train data.
        """
        feature_selection = True
        if feature_selection:  # always true
            feature_selection = FeatureSelect(self.n_features_select, self.n_csp_pairs)
            self.feature_selection = feature_selection
            x_train_features_selected = self.feature_selection.fit(x_features, y_train)
        else:
            x_train_features_selected = x_features
        self.model.fit(x_train_features_selected, y_train)
        y_predicted = self.model.predict(x_train_features_selected)
        return y_predicted


class FeatureSelect:
    def __init__(self, n_features_select=4, n_csp_pairs=2):
        self.n_features_select = n_features_select  # 'n_features_select' is the number of CSP features in each selected pair of CSP features (normally between 2-4)
        self.n_csp_pairs = n_csp_pairs  # 'n_csp_pairs' is the number of selected pairs of CSP features
        self.features_selected_indices = []

    def fit(self, x_train_features, y_train):
        """
        Select 'n_csp_pairs' pairs of CSP features, each pair contains 'n_features_select' CSP features. And save the index of selected features in 'self.features_selected_indices'.

        :param x_train_features:the CSP feature for each trial in each frequency band, each frequency band has 2*m_filters CSP features for one specific trial, (n_trials, n_fbanks*m_filters*2)
        :param y_train: the label of train data, (n_trials)
        :returns x_train_features_selected: the selected CSP features, (n_trials, n_csp_pairs*n_features_select)
        """
        MI_features = self.MIBIF(x_train_features, y_train)
        MI_sorted_idx = np.argsort(MI_features)[::-1]
        features_selected = MI_sorted_idx[:self.n_features_select]

        paired_features_idx = self.select_CSP_pairs(features_selected, self.n_csp_pairs)
        x_train_features_selected = x_train_features[:, paired_features_idx]
        self.features_selected_indices = paired_features_idx

        return x_train_features_selected

    def transform(self, x_test_features):
        return x_test_features[:, self.features_selected_indices]

    def MIBIF(self, x_features, y_labels):
        """
        Compute the mutual information for each feature with respect to the train labels.

        :param x_features: the CSP feature for each trial in each frequency band, each frequency band has 2*m_filters CSP features for one specific trial, (n_trials, n_fbanks*m_filters*2)
        :param y_labels: the labels for each trial, (n_trials)
        :returns mifsg: the mutual information for each column of CSP feature with respect to the target, (n_fbanks*m_filters*2)
        """
        def get_prob_pw(x, d, i, h):
            """
            Compute the probability when the 'i' column of given CSP feature group 'd' equals to the given value 'x' for each trial under gaussian probability density function.

            :param x: the value of the CSP feature in the index of 'i', float
            :param d: the features of trials that have one same class, (n_selected_trials, n_fbanks*m_filters*2)
            :param i: the index of chosen column from all CSP features (which has n_fbanks*m_filters*2 in total), int
            :param h: the width of Parzen window, float
            :returns prob_x: the probability when the CSP feature equals to 'x' for each trial, float
            """
            n_data = d.shape[0]  # the number of selected features
            t = d[:, i]
            kernel = lambda u: np.exp(-0.5*(u**2))/np.sqrt(2*np.pi)  # gaussian probability density function
            prob_x = 1 / (n_data * h) * sum(kernel((np.ones((len(t)))*x - t)/h))
            return prob_x

        def get_pd_pw(d, i, x_trials):
            """
            Compute the probability when the 'i' column of CSP feature group 'd' equals to each value of the 'x_trials'.

            :param d: the features of trials that have one same class, (n_selected_trials, n_fbanks*m_filters*2)
            :param i: the index of chosen column from all CSP features (which has n_fbanks*m_filters*2 in total), int
            :param x_trials: one group of CSP features of all trials in one specific frequency sub-band, (n_trials)
            :returns prob_x: the probability when the 'i' column of CSP feature group 'd' equals to each value of the 'x_trials', (n_trials)
            :returns x_trials: the same as the input 'x_trials'
            :returns h: the width of Parzen window, float
            """
            n_data, n_dimensions = d.shape
            if n_dimensions == 1:
                i = 1
            t = d[:, i]
            min_x = np.min(t)
            max_x = np.max(t)
            n_trials = x_trials.shape[0]
            std_t = np.std(t)  # standard deviation of the 'i' column of the CSP features
            if std_t == 0:
                h = 0.005
            else:
                h = (4./(3*n_data))**(0.2)*std_t
            prob_x = np.zeros((n_trials))
            for j in range(n_trials):
                prob_x[j] = get_prob_pw(x_trials[j], d, i, h)
            return prob_x, x_trials, h

        y_classes = np.unique(y_labels)
        n_classes = len(y_classes)
        n_trials = len(y_labels)
        prob_w = []
        x_cls = {}
        for i in range(n_classes):
            cls = y_classes[i]
            cls_indx = np.where(y_labels == cls)[0]
            prob_w.append(len(cls_indx) / n_trials)  # 'prob_w' is the probability for each class, (n_classes)
            x_cls.update({i: x_features[cls_indx, :]})

        prob_x_w = np.zeros((n_classes, n_trials, x_features.shape[1]))  # (n_classes, n_trials, n_fbanks*m_filters*2)
        prob_w_x = np.zeros((n_classes, n_trials, x_features.shape[1]))
        h_w_x = np.zeros((x_features.shape[1]))
        mutual_info = np.zeros((x_features.shape[1]))
        parz_win_width = 1.0 / np.log2(n_trials)
        h_w = -np.sum(prob_w * np.log2(prob_w))  # the entropy for all classes, float

        for i in range(x_features.shape[1]):
            h_w_x[i] = 0
            for j in range(n_classes):
                prob_x_w[j, :, i] = get_pd_pw(x_cls.get(j), i, x_features[:, i])[0]

        t_s = prob_x_w.shape  # (n_classes, n_trials, n_fbanks*m_filters*2)
        n_prob_w_x = np.zeros((n_classes, t_s[1], t_s[2]))  # (n_classes, n_trials, n_fbanks*m_filters*2)
        for i in range(n_classes):
            n_prob_w_x[i, :, :] = prob_x_w[i] * prob_w[i]

        prob_x = np.sum(n_prob_w_x, axis=0)  # (n_trials, n_fbanks*m_filters*2)

        for i in range(n_classes):
            prob_w_x[i, :, :] = n_prob_w_x[i, :, :]/prob_x

        for i in range(x_features.shape[1]):
            for j in range(n_trials):
                t_sum = 0.0
                for k in range(n_classes):
                    if prob_w_x[k, j, i] > 0:
                        t_sum += (prob_w_x[k, j, i] * np.log2(prob_w_x[k, j, i]))

                h_w_x[i] -= (t_sum / n_trials)

            mutual_info[i] = h_w - h_w_x[i]

        mifsg = np.asarray(mutual_info)
        return mifsg

    def select_CSP_pairs(self, features_selected, n_pairs):
        features_selected += 1
        sel_groups = np.unique(np.ceil(features_selected/n_pairs))
        paired_features = []
        for i in range(len(sel_groups)):
            for j in range(n_pairs-1, -1, -1):
                paired_features.append(sel_groups[i]*n_pairs-j)

        paired_features = np.asarray(paired_features, dtype=np.int)-1

        return paired_features
