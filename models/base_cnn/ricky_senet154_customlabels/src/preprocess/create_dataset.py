import sys
import argparse
import collections
import pickle
from pprint import pprint

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..utils import misc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    return parser.parse_args()


def show_distribution(dataset):
    counter = collections.defaultdict(int)
    for row in dataset.itertuples():
        for label in row.labels.split():
            counter[label] += 1
        if not row.labels:
            counter['negative'] += 1
        counter['all'] += 1
    pprint(counter)


def parse_position(df):
    expanded = df.ImagePositionPatient.apply(lambda x: pd.Series(x))
    expanded.columns = ['Position1', 'Position2', 'Position3']
    return pd.concat([df, expanded], axis=1)


def parse_orientation(df):
    expanded = df.ImageOrientationPatient.apply(lambda x: pd.Series(x))
    expanded.columns = ['Orient1', 'Orient2', 'Orient3', 'Orient4', 'Orient5', 'Orient6']
    return pd.concat([df, expanded], axis=1)


def add_adjacent_labels(df):
    df = df.sort_values('PositionOrd')

    records = []
    print('making adjacent labels...')
    for index,group in tqdm(df.groupby('StudyInstanceUID')):

        labels = list(group.labels)
        for j,id in enumerate(group.ID):
            if j == 0:
                left = labels[j-1]
            else:
                left = ''
            if j+1 == len(labels):
                right = ''
            else:
                right = labels[j+1]

            records.append({
                'LeftLabel': left,
                'RightLabel': right,
                'ID': id,
            })
    return pd.merge(df, pd.DataFrame(records), on='ID')


def get_softlabel(same_patient_df):
    label_to_num = {
        'any': 0,
        'epidural': 1,
        'subdural': 2,
        'subarachnoid': 3,
        'intraventricular': 4,
        'intraparenchymal': 5,
    }
    num_to_label = {
        "0": 'any',
        "1": 'epidural',
        "2": 'subdural',
        "3": 'subarachnoid',
        "4": 'intraventricular',
        "5": 'intraparenchymal',
    }

    tmp_df = same_patient_df
    label_array = np.zeros((tmp_df.shape[0], 6))
    for i, row in tmp_df.reset_index(drop=True).iterrows():
        for label in row.labels.split():
            label_array[i, label_to_num[label]] = 1

    soft_label_array = np.zeros((tmp_df.shape[0], 6))
    for i in range(6):
        idx = np.where(label_array[:, i]==1)[0]
        if len(idx) == 0:
            continue
        idx_min = idx[0]
        idx_max = idx[-1]
        soft_label_array[idx_min:idx_max+1, i] = 1
    soft_label_array = soft_label_array - label_array
    soft_label_array = soft_label_array.astype(int).astype(str)

    for i in range(6):
        soft_label_array[soft_label_array[:, i]=="1", i] = num_to_label[str(i)]
        soft_label_array[soft_label_array[:, i]=="0", i] = ""

    result_list = []
    for i in soft_label_array:
        w = ""
        for j in range(6):
            if i[j] != "":
                w = w+" "+i[j]
        result_list.append(w[1:])

    return result_list


def set_softlabes(df):
    l = []
    for uid in tqdm(df.StudyInstanceUID.unique(), total=len(df.StudyInstanceUID.unique())):
        tmp_df = df[df.StudyInstanceUID==uid]
        if not all(tmp_df.labels==""):
            tmp_df["soft_labels"] = get_softlabel(tmp_df)
        else:
            tmp_df["soft_labels"] = ""
        l.append(tmp_df)
    new_df = pd.concat(l)
    new_df.set_index("ID").loc[df.ID].reset_index().head()
    return new_df


def main():
    args = get_args()

    with open(args.input, 'rb') as f:
        df = pickle.load(f)
    print('read %s (%d records)' % (args.input, len(df)))

    show_distribution(df)

    df = df[df.custom_diff > 60]
    print('removed records by custom_diff (%d records)' % len(df))

    df = parse_position(df)

    df['WindowCenter'] = df.WindowCenter.apply(lambda x: misc.get_dicom_value(x))
    df['WindowWidth'] = df.WindowWidth.apply(lambda x: misc.get_dicom_value(x))
    df['PositionOrd'] = df.groupby('SeriesInstanceUID')[['Position3']].rank() / df.groupby('SeriesInstanceUID')[['Position3']].transform('count')

    print("set soft labels")
    df = set_softlabes(df)
    df = df[['ID', 'labels', 'PatientID', 'WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope', 'Position3', 'PositionOrd', 'soft_labels']]

    df = df.sort_values('ID')
    with open(args.output, 'wb') as f:
        pickle.dump(df, f)

    show_distribution(df)

    print('created dataset (%d records)' % len(df))
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
