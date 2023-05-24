from bin.MLEngine import MLEngine

if __name__ == "__main__":
    '''Example for loading Korea University Dataset'''
    # dataset_details = {
    #     'data_path': "/Volumes/Transcend/BCI/KU_Dataset/BCI dataset/DB_mat",
    #     'subject_id': 1,
    #     'sessions': [1],
    #     'ntimes': 1,
    #     'kfold': 10,
    #     'm_filters': 2,
    # }

    '''Example for loading BCI Competition IV Dataset 2a A01T.gdf'''
    dataset_details = {
        'data_path': "E:\\EEG\\DataSets\\BCI Competition IV\\BCICIV_2a_gdf",
        'file_to_load': 'A01T.gdf',
        'ntimes': 1,
        'kfold': 10,
        'm_filters': 2,
        'window_details': {'tmin': 1, 'tmax': 4}
    }
    ML_experiment = MLEngine(**dataset_details)

    classifier_SVR = {
        'classifier_type': 'SVR',
        'gamma': 'auto'
    }

    classifier_GaussianNB = {
        'classifier_type': 'GaussianNB',
        'var_smoothing': 1e-9
    }

    classifier_NBPW = {
        'classifier_type': 'NBPW'
    }

    classifier_kNN = {
        'classifier_type': 'kNN',
        'k': 7
    }

    MIBIF_details = {
        'feature_selection_type': 'MIBIF',
        'n_features_select': 4,
        'n_csp_pairs': 2
    }

    experiment_details = {
        'classifier': classifier_NBPW,
        'feature_selection': MIBIF_details,
    }

    ML_experiment.experiment(experiment_details)
