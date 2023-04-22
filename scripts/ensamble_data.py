import os
import pandas as pd


def ensamble():

    path = "/mnt/home/grunew14/Documents/project/data_target"
    files = os.listdir(path)
    targets_df = pd.DataFrame()
    for i, file in enumerate(files):
        df = pd.read_csv(path + "/" + file)
        targets_df = targets_df.append(df)
        targets_df.to_csv('/mnt/home/grunew14/Documents/project/data/target.csv', index=False, )


def main():
    ensamble()


if __name__ == "__main__":
    main()
