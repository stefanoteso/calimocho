from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import Experiment, load, dump



class AdultExperiment(Experiment):
    def __init__(self, **kwargs):

        # NOTE after removing rows with ?'s, some categories do not appear in
        # the data anymore

        NAMES_AND_KINDS = [
            ('age',                 'int'),
            ('workclass',           [
                'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']),
            ('fnlwgt',              'int'),
            ('education',           [
                'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']),
            ('education-num',       'int'),
            ('marital-status',      [
                'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']),
            ('occupation',          [
                'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                'Armed-Forces']),
            ('relationship',        [
                'Wife', 'Own-child', 'Husband', 'Not-in-family',
                'Other-relative', 'Unmarried']),
            ('race',                [
                'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other',
                'Black']),
            ('sex',                 ['Female', 'Male']),
            ('capital-gain',        'int'),
            ('capital-loss',        'int'),
            ('hours-per-week',      'int'),
            ('native-country',      [
                'United-States', 'Cambodia', 'England', 'Puerto-Rico',
                'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
                'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
                'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France',
                'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
                'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
                'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                'Peru', 'Hong', 'Holand-Netherlands']),
            ('label',               ['>50K', '<=50K']),
        ]
        names = [name for name, _ in NAMES_AND_KINDS]
        df_tr = pd.read_csv(join('data', 'adult.data'), names=names,
                            sep=', ', na_values=['?'], header=None).dropna()
        df_ts = pd.read_csv(join('data', 'adult.test'), skiprows=1, names=names,
                            sep=', ', na_values=['?'], header=None).dropna()
        n_train = df_tr.shape[0]

        df = pd.concat([df_tr, df_ts])
        for name, kind in NAMES_AND_KINDS:
            if kind == 'int':
                assert df[name].dtype == np.int64
            else:
                assert df[name].dtype == np.object
                df[name] = df[name].astype('category').cat.codes
                assert df[name].dtype == np.int8

        assert df.shape == (45222, 15)

        # TODO add explicit bias for SENNs

        # XXX ideally we'd use the gold standard split, for now we don't
        #X_tr, y_tr = df.values[:n_train,:-1], df.values[:n_train,-1]
        #X_ts, y_ts = df.values[n_train:,:-1], df.values[n_train:,-1]
        X, y = df.values[:,:-1], 1 - df.values[:,-1]

        # the age attribute is protected and should not be used
        Z = np.ones_like(X)
        Z[:,0] = 0

        super().__init__(X, Z, y, **kwargs)


    def dump_explanation(self, path, x, z_true, z_pred):
        fig, axes = plt.subplots(1, 2, sharey=True)

        print(z_true, z_pred)

        for i, z in enumerate([z_true, z_pred]):
            axes[i].set_aspect('equal')

            z = z.reshape(-1, 1)
            axes[i].imshow(z, interpolation='nearest', cmap='binary')

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)
