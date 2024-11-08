#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:19:14 2023

@author: anonymous
"""

import argparse
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
from prdc import compute_prdc
from scipy import stats
from scipy.spatial import distance
from statsmodels.tsa.stattools import acf
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def compute_WD(real, generated):#EMD in paper
    score = 0
    for i in real.columns:
        #compute the empirical cdfs
        cdf1,bins1 = np.histogram(real[i].to_numpy(), bins='fd')
        ecdf1 = np.cumsum(cdf1)/len(real)
        ecdf2 = np.cumsum(np.histogram(generated[i].to_numpy(), bins=bins1)[0])/len(generated)
        score += stats.wasserstein_distance(ecdf1, ecdf2)
    return score/len(real.columns)


def compute_JSD(df1, df2):
    #get set of all variables
    variables = set(df1.columns).union(set(df2.columns))

    #initialize empty list to store jsd values for each variable
    jsd_values = []

    #loop through variables
    for var in variables:
        #get the set of values for the variable from both dataframes
        values1 = set(df1[var].unique())
        values2 = set(df2[var].unique())

        #create a union of the sets of values for the variable
        all_values = values1.union(values2)

        #fill missing values with 0
        data1 = df1[var].value_counts().reindex(all_values, fill_value=0)
        data2 = df2[var].value_counts().reindex(all_values, fill_value=0)
        #compute jsd for the variable and append to list
        jsd = distance.jensenshannon(data1, data2)
        jsd_values.append(jsd)

    return np.mean(jsd_values)

def compute_PCD(real, generated):

    temp = real.to_numpy()
    g = generated.to_numpy()
    scaler = MinMaxScaler().fit(np.concatenate((temp, g)))#min-max normalization

    temp = scaler.transform(temp)
    g = scaler.transform(g)
    pcd_r = np.corrcoef(temp.T)
    pcd_g = np.corrcoef(g.T)
    pcd_r[np.isnan(pcd_r)] = 0#replace nan by 0
    pcd_g[np.isnan(pcd_g)] = 0
    return np.linalg.norm(pcd_r - pcd_g)

def compute_CMD(real, generated):
    #get all pairs of columns possible
    all_pairs = [(real.columns[i], real.columns[j]) for i in range(real.shape[1]) for j in range(i + 1, real.shape[1])]
    s=0
    for i in all_pairs:
        contingency_table_r = pd.crosstab(real[i[0]], real[i[1]], dropna=False, normalize=True)
        contingency_table_g = pd.crosstab(generated[i[0]], generated[i[1]], dropna=False, normalize=True)

        #list of all unique values in both variables
        all_categories_0 = sorted(set(real[i[0]].unique()).union(generated[i[0]].unique()))
        all_categories_1 = sorted(set(real[i[1]].unique()).union(generated[i[1]].unique()))

        #extend the contingencie tables with all the possible values
        contingency_table_r_extended = pd.DataFrame(index=all_categories_0, columns=all_categories_1)
        contingency_table_r_extended.update(contingency_table_r)
        contingency_table_g_extended = pd.DataFrame(index=all_categories_0, columns=all_categories_1)
        contingency_table_g_extended.update(contingency_table_g)
        #fill missing values with 0
        contingency_table_r_extended = contingency_table_r_extended.fillna(0)
        contingency_table_g_extended = contingency_table_g_extended.fillna(0)

        s+= np.linalg.norm(contingency_table_r_extended - contingency_table_g_extended)
    return s/len(all_pairs)

def transform_discretize(df, i=50):
    #this function discretize feature that does not have sufficent distinct value to be considere continuosu
    for c in df.columns:
        if df[c].nunique()>i and 'IP' not in c and 'Pt' not in c and 'Flags' not in c:
            df[c] = pd.qcut(df[c],i,duplicates='drop')
    return df

def transform_OHE(df, i=50):
    #this function get the one hot encoding of all the columns
    for col in df.columns:
        if df[col].nunique()<i or 'IP' in col or 'Pt' in col or 'Flags' in col:
            df = pd.concat([df,pd.get_dummies(df[col],prefix=col+'_is', prefix_sep='_')],axis=1)
            df=df.drop(col,axis=1)
    return df

def compute_authenticity(train, test, generated, i= 50, n = 500):#MD in the paper
    temp = transform_discretize(pd.concat([train, test, generated]), i)#Turn every feature discrete

    u= []

    for c in temp.columns:
        u.extend(list(temp[c].unique()))#get the list and index of al unique value
    u = list(set(u))
    tr = train.replace(dict(zip(u, list(range(len(u))))))
    ts = test.replace(dict(zip(u, list(range(len(u))))))
    g = generated.replace(dict(zip(u, list(range(len(u))))))
    #g is a dataset of integer
    ts = ts.sample(n).to_numpy() ##test set
    tr = tr.sample(n).to_numpy() ##train set
    g = g.sample(n).to_numpy() ##generated set

    M = np.ones((len(ts)+len(tr), len(g)))
    for i, row in enumerate(np.concatenate([ts, tr])):
        for j, col in enumerate(g):
            M[i, j] = distance.hamming(row, col)#calculate hamming distance beetwen all the generated samples an all the real (train+test) samples.
    score = 0
    for r in np.linspace(0,1,15):#for every r
        u = M <= r#True the hamming distance between a real sample and a generated sample is lower than R
        result = (np.count_nonzero(u, axis=1) > 0)#See the real sample that has a hamming distance to a generated sample inferior to r
        label = np.concatenate([np.zeros(len(ts)), np.ones(len(tr))]).astype(bool)#We know wich label are training and wich is not
        if result.sum() == 0:
            continue

        pr = np.logical_and(result, label).sum()/label.sum()
        rr = np.logical_and(result, label).sum()/result.sum()
        f1= 2*pr*rr/(pr+rr)

        score += f1#Score is the summation of the f1 for all the r
    return score

def compute_density_coverage(real, g, i=40, n=5):
    temp = transform_OHE(pd.concat([real, g]), i)

    temp = temp.astype(float)

    r = temp.head(len(real)).to_numpy()
    generated = temp.tail(len(g)).to_numpy()

    scaler = MinMaxScaler().fit(np.concatenate((r, generated)))

    r = scaler.transform(r)
    generated = scaler.transform(generated)
    scores = list(compute_prdc(r, generated, n).values())
    return tuple(scores[-2:])

def compute_DKC(u):
    score = 0
    generated = u.astype(str)
    score+=len(generated[((generated["Dst Pt"].isin(['53.0', '137.0', '138.0', '5353.0', '1900.0', '67.0', '0.0', '3544.0', '8612.0', '3702.0', '123.0'])) & (generated["Proto"].str.contains("TCP")))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(['443.0', '80.0', '8000.0', '25.0', '993.0', '587.0', '445.0', '0.0', '84.0', '8088.0', '8080.0'])) & (generated["Proto"].str.contains("UDP")))])/len(generated)
    score+=len(generated[((generated["Dst Pt"] == "0.0") & ((~generated["Proto"].str.contains("ICMP")) | (~generated["Proto"].str.contains("IGMP"))))])/len(generated)
    score+=len(generated[((generated["Proto"].str.contains("ICMP")) & (generated["Out Byte"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Proto"].str.contains("ICMP")) & (generated["Out Packet"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Proto"].str.contains("IGMP")) & ((generated["Out Byte"]!="0.0") | (generated["In Byte"]!="0.0")))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(["137.0", "138.0", "1900.0"])) & (generated["In Byte"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(["137.0", "138.0", "1900.0"])) & (generated["In Packet"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(["137.0", "138.0", "1900.0"])) & (~generated["Dst IP Addr"].str.endswith(".255")))])/len(generated)
    score+=len(generated[(generated["Dst Pt"].isin(["8000.0", "25.0", "443.0","80","587"])) & (generated["Dst IP Addr"].str.startswith("192.168"))])/len(generated)
    score+=len(generated[(generated["Dst Pt"].isin(["993.0", "67.0"])) & (~generated["Dst IP Addr"].str.startswith("192.168"))])/len(generated)
    score+=len(generated[(generated["Dst Pt"]=="53.0") & (generated["Dst IP Addr"] != "DNS")])/len(generated)
    score+=len(generated[(generated["Dst Pt"]=="5353.0") & (generated["Dst IP Addr"] != "10008_251")])/len(generated)
    score+=len(generated[(generated["Flags"]!="......") & (generated["Proto"]!="TCP")])/len(generated)
    score+=len(generated[generated["In Packet"].astype(float)*42 > generated["In Byte"].astype(float)])/len(generated)
    score+=len(generated[generated["Out Packet"].astype(float)*42 > generated["Out Byte"].astype(float)])/len(generated)
    score+=len(generated[generated["In Byte"].astype(float) > 65535*generated["In Packet"].astype(float)])/len(generated)
    score+=len(generated[generated["Out Byte"].astype(float) > 65535*generated["Out Packet"].astype(float)])/len(generated)
    score+=len(generated[generated["Duration"].str.contains("-")])/len(generated)
    score+=len(generated[((generated["Duration"].astype(float)==0) & (generated["In Packet"].astype(float)+generated["Out Packet"].astype(float)>1))])/len(generated)
    score+=len(generated[((generated["Duration"].astype(float)>00) & (generated["In Packet"].astype(float)+generated["Out Packet"].astype(float)==1))])/len(generated)
    return score/20

def compare_corr_accross_time(real, generated, t):
    #this function will compare correlation between different timesteps
    #we will have two correlation matrices, one for the real data, and one for the generated
    s=0
    for i in range(real.shape[1]):
        corr_coef=[]
        for df in [real, generated]:
            u = df.iloc[:-t,i].reset_index(drop=true) #initial timestep
            v = df.iloc[t:,i].reset_index(drop=true) #shifted from t timesteps
            j = pd.concat([u, v],axis=1)
            corr = np.corrcoef(j.t) #correlation matrix beetwen attributes at timestep n and at timestep n + t
            corr_coef.append(corr[0,1])
        s += np.linalg.norm(corr_coef[1]-corr_coef[0])/corr_coef.shape[0]#l2 norm difference beetwen correlation matrix of real data and correlation matrix of fake data
        if np.isnan(s):
            return 0
    return s/real.shape[1]

def compare_acf_timestep(real, generated, alpha = .05):
    ''' This function will compare the autocorrelation function (acf) of the Real and the Generated timeseries
        We use for this the statsmodels library
        We first compute the acf of the Real timeserie with the confidence interval
        We then determine for wich lag t, the autocorellation at lag t is statistically significant (Bartlett formula)
        And then we compare autocorelation for Real and Generated on that specific t'''
    if len(real)<2 or len(generated)<2:
        return np.nan
    real_corr_coeffs, real_barts_range = acf(real, nlags=len(real)-1, alpha = alpha, bartlett_confint=True, qstat=False)

        #we are looking for the lags for wich 0 is exluded from the error margin
    real_lags = np.arange(len(real))[real_barts_range[:,0]*real_barts_range[:,1]>=0]

        #same for the generated
    gen_corr_coeffs, gen_barts_range = acf(generated, nlags=len(generated)-1, alpha = alpha, bartlett_confint=True, qstat=False)

    real_lags = real_lags[~np.isnan(real_lags)]
    if len(real_lags)>0:
        gen_corr_coeffs = np.pad(gen_corr_coeffs, (0, max(0, 1 + max(real_lags) - len(gen_corr_coeffs))), 'constant')

    real_corr_coeffs = np.nan_to_num(real_corr_coeffs)
    gen_corr_coeffs = np.nan_to_num(gen_corr_coeffs)

        #lags = np.unique(np.concatenate([real_lags,gen_lags]))
    lags = np.unique(real_lags)
        #we want to mesure the distance beetwen correlations beetwen the two datasets
    s = np.sum(np.abs(real_corr_coeffs[lags] - gen_corr_coeffs[lags])) / len(real)
    return s

def compare_acf_timerange(real, generated):
    #this function will do compare_acf_timestep, but with timestep of 1 second.
    rr=real.resample("s").sum()#Sum all the value for each second
    gg=generated.resample("s").sum()
    common_idx = rr.index.intersection(gg.index)#We consider only the intersection
    rr = rr.loc[common_idx]
    gg = gg.loc[common_idx]
    return compare_acf_timestep(rr, gg)

def compare_LSTM_discrete_feature(real, generated, num_epochs=10, batch_size=4, seq_len=64):
    #assuming the presence of a device configuration (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode_series(series, label_encoder):
        encoded = label_encoder.transform(series)
        one_hot = np.eye(len(label_encoder.classes_))[encoded]
        return one_hot

    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, encoded_data, seq_len):
            self.encoded_data = encoded_data
            self.seq_length = seq_len

        def __len__(self):
            return len(self.encoded_data) - self.seq_length

        def __getitem__(self, idx):
            return (self.encoded_data[idx:idx+self.seq_length],
                    self.encoded_data[idx+self.seq_length])

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers, learning_rate=0.001):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.learning_rate = learning_rate

            #lstm layer
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            #fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)

            #loss and optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def forward(self, x):
            #initialize hidden and cell states
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

            #forward propagate the lstm
            out, _ = self.lstm(x, (h0.detach(), c0.detach()))
            #decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out

        def train_model(self, train_loader, num_epochs):
            for epoch in range(num_epochs):
                self.train()  #set the model to training mode
                for i, (sequences, labels) in enumerate(train_loader):
                    sequences = sequences.to(device)
                    labels = labels.to(device)

                    #forward pass
                    outputs = self(sequences)
                    loss = self.criterion(outputs, labels)

                    #backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if (i+1) % 100 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        def evaluate_model(self, test_loader):
            self.eval()  #set the model to evaluation mode
            with torch.no_grad():
                correct = 0
                total = 0
                for sequences, labels in test_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    outputs = self(sequences)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == torch.argmax(labels, 1)).sum().item()
            return  correct / total

    def create_dataloader(ser):
        #create input sequences (x) and targets (y)
        ser = torch.tensor(ser, dtype=torch.float32)

        train_dataset = SequenceDataset(ser, seq_len)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader

    def get_the_score_of_a_column(real, generated):
        #encode your categories
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([real, generated])) #fit on all possible values

        real_encoded = encode_series(real,label_encoder)
        generated_encoded = encode_series(generated,label_encoder)

        real_loader = create_dataloader(real_encoded)
        generated_loader = create_dataloader(generated_encoded)

        #create the model
        model = LSTMModel(
            input_dim = len(label_encoder.classes_),
            hidden_dim = 50,
            output_dim = len(label_encoder.classes_),
            num_layers = 2,
            learning_rate = 0.001
        ).to(device)

        model.train_model(real_loader, num_epochs)

        score = model.evaluate_model(generated_loader)

        return score

    score = 0

    for col in real.columns:
        t = get_the_score_of_a_column(real[col], generated[col])
        print(col, t)
        score +=t

    return score

def LSTM_discrete_dataset(train, generated, test, num_epochs=7, batch_size=4, seq_len=64, hidden_dim = 300, tstr=True):
    #assuming the presence of a device configuration (cpu/gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def encode_series(series, label_encoder):
        encoded = label_encoder.transform(series)
        return torch.tensor(encoded, dtype=torch.long)

    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, encoded_data, seq_len):
            self.encoded_data = encoded_data
            self.seq_length = seq_len

        def __len__(self):
            return len(self.encoded_data) - self.seq_length

        def __getitem__(self, idx):
            return (self.encoded_data[idx:idx+self.seq_length],
                    self.encoded_data[idx+self.seq_length])

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers, learning_rate=0.001, dtype=torch.float32, device=device):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.learning_rate = learning_rate
            self.device=device
            self.dtype = dtype  #data type for model weights and computations

            #embedding layer
            self.embedding = nn.Embedding(input_dim, hidden_dim).to(dtype=self.dtype, device=self.device)
    
            #lstm layer
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.lstm = self.lstm.to(dtype=self.dtype, device=self.device)
    
            #fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.fc = self.fc.to(dtype=self.dtype, device=self.device)
    
            #loss and optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
        def forward(self, x):
            #initialize hidden and cell states with the specified dtype
            x = self.embedding(x)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=self.dtype, device=self.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=self.dtype, device=self.device)
    
            #forward propagate the lstm
            out, _ = self.lstm(x, (h0.detach(), c0.detach()))
            #decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out
    
        def train_model(self, train_loader, num_epochs):
            for epoch in range(num_epochs):
                self.train()  #set the model to training mode
                for i, (sequences, labels) in enumerate(train_loader):
                    sequences = sequences.to(device=self.device)
                    labels = labels.to(device=self.device)
    
                    #forward pass
                    outputs = self(sequences)
                    loss = self.criterion(outputs, labels)
    
                    #backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
    
                    if (i + 1) % 10 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
        def evaluate_model(self, test_loader):
            self.eval()  #set the model to evaluation mode
            with torch.no_grad():
                correct = 0
                total = 0
                for sequences, labels in test_loader:
                    sequences = sequences.to(device=self.device)
                    labels = labels.to(device=self.device)
    
                    outputs = self(sequences)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    
                accuracy = correct / total
                return accuracy
                
    def create_dataloader(ser):

        #create input sequences (x) and targets (y)
        train_dataset = SequenceDataset(ser, seq_len)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader

    label_encoder = LabelEncoder()
    df = pd.concat([train, test, generated])
    df = pd.concat([df,df],axis=1)
    df.columns = ["orig","combined"]
    label_encoder.fit(df['combined']) #fit on all possible values

    real_encoded = encode_series(df.iloc[:len(train),-1],label_encoder)
    real_loader = create_dataloader(real_encoded)
    del real_encoded
    
    ts_encoded = encode_series(df.iloc[len(train):-len(generated),-1],label_encoder)
    ts_loader = create_dataloader(ts_encoded)
    del ts_encoded
    
    generated_encoded = encode_series(df.iloc[-len(generated):,-1],label_encoder)
    generated_loader = create_dataloader(generated_encoded)
    del generated_encoded

    del df

        #create the model
    model = LSTMModel(
            input_dim = len(label_encoder.classes_),
            hidden_dim = hidden_dim,
            output_dim = len(label_encoder.classes_),
            num_layers = 2,
            learning_rate = 0.001,
            dtype = torch.float32,
            device=device
    )
    
    if not tstr:
        print("training begin")
        model.train_model(real_loader, num_epochs)
    
        score = model.evaluate_model(generated_loader)
    
    if tstr:
        print(f"Doing the LSTM comparison. LSTM parameters: hidden_dim - {hidden_dim}; seq_len - {seq_len}")
        model.train_model(real_loader, num_epochs)
    
        score_r = model.evaluate_model(ts_loader)

        model.train_model(generated_loader, num_epochs)

        score_g = model.evaluate_model(ts_loader)

        score = score_r - score_g
        
    return np.abs(score)

if __name__ == '__main__':
    path = "../data/"
    print("Loading dataframes")
    train = pd.read_csv(path+"train.csv")
    test = pd.read_csv(path+"test.csv")
    test["Date first seen"] = pd.to_datetime(test["Date first seen"])-pd.to_timedelta(7*3600*24,unit="s")

    ewgangp = pd.read_csv(path+"ewgan-gp_syn.csv")
    ctgan = pd.read_csv(path+"ctgan_syn.csv")
    tvae = pd.read_csv(path+"tvae_syn.csv")
    transformer = pd.read_csv(path+"transformer_syn.csv")
    netshare = pd.read_csv(path+"netshare_syn.csv")
    bn_indep = pd.read_csv(path+"BN_baseline_window0_syn.csv")
    bn_timdep = pd.read_csv(path+"BN_baseline_window5_syn.csv")
    our = pd.read_csv(path+"our_syn.csv")

    models = ['Simulation', 'IndependantBN', 'SequenceBN', 'TVAE', 'CTGAN' ,'E-WGAN-GP', 'NetShare', 'Transformer', 'FlowChronicle']

    continuous = ['In Byte', 'Out Byte', 'In Packet', 'Out Packet', 'Duration']
    discrete_1 = ['Proto', 'Src IP Addr', 'Dst IP Addr', 'Dst Pt', 'Flags']
    
    ts = pd.read_csv(path+"/CIDDS_xp2_evaluate.csv")

    datasets = [train, test, bn_indep, bn_timdep, tvae, ctgan, ewgangp, netshare, transformer, our, ts]

    for i in range(len(datasets)):
        datasets[i]["Date first seen"] = pd.to_datetime(datasets[i]['Date first seen'])
        datasets[i] = datasets[i].sort_values("Date first seen")
        datasets[i]["Proto"] = datasets[i]["Proto"].str.strip()
        datasets[i].index=datasets[i].pop("Date first seen")
        datasets[i] = datasets[i][discrete_1].astype(str)
    print("Finished loading dataframes")
    score_tstr = np.zeros((len(models),len(discrete_1), 9))

    real = datasets.pop(0)
    ts = datasets.pop(-1)
    
    batch_size = 4000

    for i, m in enumerate(datasets):
        for j, c in enumerate(discrete_1):
            for k, hidden_dim in enumerate([500, 400, 300]):
                for l, seq_len in enumerate([256, 128, 64]):
                    score_tstr[i,j, 3*k+l] = LSTM_discrete_dataset(real[c], m[c], ts[c], batch_size=batch_size, hidden_dim=hidden_dim, seq_len=seq_len, tstr = True)
        print(models[i], score_tstr[i,:,:])

    avg = pd.DataFrame(np.mean(score_tstr, axis=-1), columns=discrete_1,index=models)
    print(avg)
    avg.to_csv("../results/temporal_dep_discrete_avg.csv")
