import random
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader


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

def trans_tuple(input_list):
    indexed_data = [(i, item) for i, sublist in enumerate(input_list, start=1) for item in sublist]
    df = pd.DataFrame(indexed_data, columns=['Index', 'Data'])
    df[['Category', 'Aspect', 'Sentiment', 'Opinion']] = pd.DataFrame(df['Data'].tolist(), index=df.index)
    df.drop('Data', axis=1, inplace=True)
    df.rename(columns={'Index': 'sent_id'}, inplace=True)
    df['quad_ord'] = df.groupby('sent_id').cumcount() + 1
    df['max_ord'] = df.groupby('sent_id')['quad_ord'].transform('max').astype(int)
    df = df[['sent_id', 'quad_ord', 'max_ord', 'Category', 'Aspect', 'Opinion', 'Sentiment']]
    return df


def extract_quad(quard_list, seq_type='gold'):
    target = []
    for seq in quard_list:
        quads = []
        sents = [s.strip() for s in seq.split('[SSEP]')]
        for s in sents:
            try:
                tok_list = ["[C]", "[S]", "[A]", "[O]"]
    
                for tok in tok_list:
                    if tok not in s:
                        s += " {} null".format(tok)
                index_ac = s.index("[C]")
                index_sp = s.index("[S]")
                index_at = s.index("[A]")
                index_ot = s.index("[O]")
    
                combined_list = [index_ac, index_sp, index_at, index_ot]
                arg_index_list = list(np.argsort(combined_list))
    
                result = []
                for i in range(len(combined_list)):
                    start = combined_list[i] + 4
                    sort_index = arg_index_list.index(i)
                    if sort_index < 3:
                        next_ = arg_index_list[sort_index + 1]
                        re = s[start:combined_list[next_]]
                    else:
                        re = s[start:]
                    result.append(re.strip())
    
                ac, sp, at, ot = result
    
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'null'
            except ValueError:
                try:
                    print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''
    
            quads.append((ac, at, sp, ot))
    
        target.append(quads)
    return target

def error_type(predict_df, target_df):

    t_f = target_df
    p_f = predict_df
    merged_df =[]

    print("init: target=",len(t_f)," predict=",len(p_f))
    # all_ match
    all_match = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t', 'Aspect_t', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Category_p', 'Aspect_p', 'Opinion_p', 'Sentiment_p'])
    all_match = all_match.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    all_match['score']='all_match'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(all_match.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(all_match.set_index(['sent_id', 'quad_ord_p']).index)]
    print("all_match:",len(all_match)," + target",len(t_f)," = ",len(all_match) + len(t_f),"  :: predict=",len(p_f))
    
    #C_error
    C_error = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Aspect_p', 'Opinion_p', 'Sentiment_p'])
    C_error = C_error.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    C_error['score']='C_error'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(C_error.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(C_error.set_index(['sent_id', 'quad_ord_p']).index)]
    print("C_error:",len(C_error)," + target",len(t_f)," = ",len(C_error) + len(t_f),"  :: predict=",len(p_f))
    
    #A_error
    A_error = pd.merge(t_f, p_f, left_on=['sent_id', 'Category_t', 'Opinion_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Category_p', 'Opinion_p', 'Sentiment_p'])
    A_error = A_error.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    A_error['score']='A_error'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(A_error.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(A_error.set_index(['sent_id', 'quad_ord_p']).index)]
    print("A_error:",len(A_error)," + target",len(t_f)," = ",len(A_error) + len(t_f),"  :: predict=",len(p_f))
    
    #O_error
    O_error = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Category_t', 'Sentiment_t'],
                         right_on=['sent_id', 'Aspect_p', 'Category_p', 'Sentiment_p'])
    O_error = O_error.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    O_error['score']='O_error'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(O_error.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(O_error.set_index(['sent_id', 'quad_ord_p']).index)]
    print("O_error:",len(O_error)," + target",len(t_f)," = ",len(O_error) + len(t_f),"  :: predict=",len(p_f))
    
    #S_error
    S_error = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Opinion_t', 'Category_t'],
                         right_on=['sent_id', 'Aspect_p', 'Opinion_p', 'Category_p'])
    S_error = S_error.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    S_error['score']='S_error'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(S_error.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(S_error.set_index(['sent_id', 'quad_ord_p']).index)]
    print("S_error:",len(S_error)," + target",len(t_f)," = ",len(S_error) + len(t_f),"  :: predict=",len(p_f))
    
    #AO_Gold
    AO_Gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t', 'Opinion_t'],
                         right_on=['sent_id', 'Aspect_p', 'Opinion_p'])
    AO_Gold = AO_Gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    AO_Gold['score']='AO_Gold'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(AO_Gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(AO_Gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("AO_Gold:",len(AO_Gold)," + target",len(t_f)," = ",len(AO_Gold) + len(t_f),"  :: predict=",len(p_f))
    
    #A_Gold
    A_Gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Aspect_t'],
                         right_on=['sent_id', 'Aspect_p'])
    A_Gold = A_Gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    A_Gold['score']='A_Gold'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(A_Gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(A_Gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("A_Gold:",len(A_Gold)," + target",len(t_f)," = ",len(A_Gold) + len(t_f),"  :: predict=",len(p_f))
    
    #O_Gold
    O_Gold = pd.merge(t_f, p_f, left_on=['sent_id', 'Opinion_t'],
                         right_on=['sent_id', 'Opinion_p'])
    O_Gold = O_Gold.drop_duplicates(subset=['sent_id', 'quad_ord_t'])
    O_Gold['score']='O_Gold'
    t_f = t_f[~t_f.set_index(['sent_id', 'quad_ord_t']).index.isin(O_Gold.set_index(['sent_id', 'quad_ord_t']).index)]
    p_f = p_f[~p_f.set_index(['sent_id', 'quad_ord_p']).index.isin(O_Gold.set_index(['sent_id', 'quad_ord_p']).index)]
    print("O_Gold:",len(O_Gold)," + target",len(t_f)," = ",len(O_Gold) + len(t_f),"  :: predict=",len(p_f))
    
    #etc_error
    t_f['score']='no_match'
    p_f['score']='p_error'
    
    merged_df = pd.concat([all_match, A_error, C_error, O_error, S_error, AO_Gold, A_Gold, O_Gold, t_f, p_f ])
    merged_df = merged_df.sort_values(by=['sent_id', 'quad_ord_t'])

    return merged_df