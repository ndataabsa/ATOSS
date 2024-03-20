import random
import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, DatasetDict
import json

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def split_sharp(file_name):
    inputs, targets =[], []
    with open(file_name, 'r', encoding='UTF-8') as fp:
        for line in fp:
            input, target = [], []
            input, target = line.strip().split('####')
            if line != '':
                inputs.append(input)
                targets.append(target)
    print('Data read. Total count: ',len(targets))
    return inputs, targets

def merge_sharp_n(inputs, targets, num=1):
    # 각 target 항목을 num 번씩 복사하여 확장
    targets_expanded = [tar for tar in targets for _ in range(num)]

    # 병합
    merged_data = [inp + "####" + tar for inp, tar in zip(inputs, targets_expanded)]
    # 파일로 저장
    print('Input count:', len(inputs))
    print('Expanded target count:', len(targets_expanded))
    print('Merged data count:', len(merged_data))
    print('Data return. Total count:', len(merged_data))

    return merged_data

def f1_compute(dir, xlsx_name, file_name, start=0, end=11):
    # split_xlsx_name = xlsx_name.split("{i}")
    # split_file_name = file_name.split("{i}")
    for i in range(start, end):
        
        file_path = os.path.join(dir,f'{xlsx_name[0]}{i}{xlsx_name[1]}')
        df = pd.read_excel(file_path)
        df_max = df.groupby('sent_id').agg({'max_ord_t': 'max','max_ord_p': 'max'}).reset_index()
        df_allmatch = df[df['score']=='all_match'].groupby('sent_id').agg({'score': 'count'}).reset_index()    
        merged_df = pd.merge(df_max, df_allmatch, on='sent_id', how='left')
        merged_df['score'] = merged_df['score'].fillna(0)
        merged_df['precision'] = merged_df['score']/merged_df['max_ord_p']
        merged_df['recall'] = merged_df['score']/merged_df['max_ord_t']
        merged_df['f1'] = 2 * merged_df['precision'] * merged_df['recall'] / (merged_df['precision'] + merged_df['recall'])
        merged_df['f1'] = merged_df['f1'].fillna(0)
        merged_df.rename(columns={'f1': f'f1_{i}'}, inplace=True)
        if i == start:
            f_df = merged_df[['sent_id',f'f1_{i}']]
        else:
            f_df[f'f1_{i}'] = merged_df[f'f1_{i}']
    f1_columns = [f'f1_{i}' for i in range(start, end)]
    f_df['source'] = f_df[f1_columns].idxmax(axis=1)
    f_df['source'] = f_df['source'].replace('f1_','', regex=True)
    f_df['source'] = f_df.apply(lambda row: f'{file_name[0]}{row["source"]}{file_name[1]}', axis=1)
    f_list = list(f_df['source'])

    return f_df, f_list

def find_sent_name(dir, file_names):  
    lines_list = []
    for index, file_name in enumerate(file_names):
        try:
            file_name = os.path.join(dir, file_name)
            with open(file_name, 'r') as file:
                # 파일에서 n번째 라인 읽기 (인덱스는 0부터 시작하므로, 1을 더함)
                for i, line in enumerate(file, 1):
                    if i == index + 1:
                        input, _ = line.strip().split('####')
                        lines_list.append(input)  # strip()을 사용하여 양쪽의 공백 및 개행문자 제거
                        break
        except FileNotFoundError:
            print(f'File not found: {file_name}')
            lines_list.append(None)  # 파일이 없을 경우 리스트에 None 추가
    print('Data find. Total count: ',len(lines_list))
    return lines_list

def find_sent_lines(file_path, line_numbers):
    lines_extracted = []

    with open(file_path, 'r', encoding='UTF-8') as file:
        for i, line in enumerate(file, start=1):
            if i in line_numbers:
                input, _  = line.rstrip('\n').split('####')
                lines_extracted.append(input)
    
    print('Data find. Total count: ',len(lines_extracted))
       
    return lines_extracted

def format_data(input_string):
    parts = input_string.split('####')
    prompt = parts[0].strip() if len(parts) > 0 else ""
    chosen = parts[1].strip() if len(parts) > 1 else ""
    rejected = parts[2].strip() if len(parts) > 2 else ""
    return {
        "chosen": chosen,
        "rejected": rejected,
        "prompt": prompt

    }

def process_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            formatted_data = format_data(line.strip())
            data.append(formatted_data)
    return data

def create_dataset(file_path1, file_path2):
    data1 = process_file(file_path1)
    data2 = process_file(file_path2)
    data_df1 = Dataset.from_pandas(pd.DataFrame(data1))
    data_df2 = Dataset.from_pandas(pd.DataFrame(data2))
    dataset_dict = DatasetDict({
        'train': data_df1,
        'dev': data_df2
    })

    return dataset_dict
