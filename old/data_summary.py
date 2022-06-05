import pandas as pd

def count_unique(df: pd.DataFrame) -> dict:
    unique_vals = {}
    for field in df:
        unique_vals[field] = df[field].value_counts().to_dict()
    return unique_vals

def main(csv_path, summary_stats_path):
    train_data_df = pd.read_csv(csv_path)

    unique_vals = count_unique(train_data_df)
    survivors = sum(train_data_df['Survived'] == 1)
    total_passengers = train_data_df.shape[0]

    with open(summary_stats_path, 'w') as f:
        f.write("columns = " + str(train_data_df.columns.values))
        f.write('\n')
        f.write(train_data_df.describe().to_string() + '\n')
        f.write(f'\n survival rate = {100*survivors/total_passengers:.2f}\% \n')

        for key, value in unique_vals.items():
            if len(value) < 10:
                f.write(f'\n{key}:{value}')
        for key, value in unique_vals.items():
            if len(value) >= 10:
                f.write(f'\n{key}: # unique values = {len(value)}')

if __name__ == '__main__':
    main()