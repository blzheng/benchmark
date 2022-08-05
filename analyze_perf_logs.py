import os
import sys
import pandas as pd

def list_dir(file_dir):
    file_list = os.listdir(file_dir)
    return file_list

def read_csv(log_dir, file):
    col_names = ['filename', 'BS', 'NNC_ENABLED', 'Throughput']
    csv_df = pd.read_csv(os.path.join(log_dir, file), skiprows=1, \
                             names=col_names)
    return csv_df

def calculate_ratio(csv_df):
    filename_unique = csv_df['filename'].unique()
    NNC_ENABLED = csv_df['NNC_ENABLED'].unique()
    # for file in filename_unique:
    #     perf_df = csv_df[csv_df['filename'] == file]
    #     #print(perf_df[csv_df['NNC_ENABLED'] == NNC_ENABLED[1]]['Throughput'].values)
    #     a = perf_df[csv_df['NNC_ENABLED'] == NNC_ENABLED[1]]['Throughput'].values
    #     b = perf_df[csv_df['NNC_ENABLED'] == NNC_ENABLED[0]]['Throughput'].values
    #     perf_ratio = a / b
    #     print(perf_ratio)
    # NNC_ENABLED[1] is True, 0 is False
    nnc_true = csv_df[csv_df['NNC_ENABLED'] == NNC_ENABLED[1]]['Throughput'].values
    nnc_false = csv_df[csv_df['NNC_ENABLED'] == NNC_ENABLED[0]]['Throughput'].values
    perf_ratio = nnc_true / nnc_false
    return perf_ratio.mean()

def generate_ratio_df(log_dir, log_file_list):
    ratio_df = pd.DataFrame(columns=['len', 'pattern', 'GEOMEAN'], index=[])
    for file in log_file_list:
        csv_df = read_csv(log_dir, file)
        ratio = calculate_ratio(csv_df)
        filename_split = file.split('_perf')[0].split('_')
        size = ratio_df.index.size
        ratio_df.loc[size] = [filename_split[0], filename_split[1:], ratio]
    return ratio_df

if __name__ == '__main__':
    log_dir=sys.argv[1]
    log_file_list = list_dir(file_dir=log_dir)
    ratio_df = generate_ratio_df(log_dir, log_file_list)
    ratio_df.to_csv(os.path.join(log_dir, 'summary.log'), index=False)