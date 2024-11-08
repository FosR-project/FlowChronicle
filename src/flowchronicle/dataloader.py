import json
import pickle
import pandas as pd
import numpy as np
import pandas.api.types as ptypes
from bidict import bidict
from copy import deepcopy
from flowchronicle import domain_knowledge
from flowchronicle.preprocess import optimize_gmm, bytes_to_int, time_to_int
from sklearn.mixture import GaussianMixture



flag_map = {
        'F': 'FIN',
        'S': 'SYN',
        'R': 'RST',
        'P': 'PSH',
        'A': 'ACK',
        'U': 'URG',
        'E': 'ECE',
        'C': 'CWR'
    }
class ContinousRepr:
    '''
    To real Byte value:
    random.uniform(cutpoints['In Byte][discrete_repr], cutpoints['In Byte'][discrete_repr+1])
    '''
    def __init__(self) -> None:
        self.__cutpoints = None
        self.__time_precision = None
        self.__first_flow_time = None

    def add_cutpoints(self, cutpoints):
        self.__cutpoints = cutpoints

    def get_cutpoints(self):
        return self.__cutpoints

    def add_first_flow_time(self, first_flow_time):
        self.__first_flow_time = first_flow_time

    def get_first_flow_time(self):
        return self.__first_flow_time

    def add_time_precision(self, time_precision):
        self.__time_precision = time_precision

    def get_time_precision(self):
        return self.__time_precision

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Dataset:
    def __init__(self, dataset:pd.DataFrame, bytes_dic_name="discretize_byte_stats.json"):
        if dataset is not None:
            self.initialize(dataset, bytes_dic_name)
    def initialize(self, dataset: pd.DataFrame, bytes_dic_name="discretize_byte_stats.json"):
        assert ptypes.is_integer_dtype(dataset.iloc[:,0]) # first column has to integer, assumed to be type stamps
        # assert that first column is sorted
        assert np.all(np.diff(dataset.iloc[:,0]) >= 0)

        self.__dataset = dataset
        self.column_value_dict = {}
        self.column_value_dict_str = {}
        for col in self.__dataset.columns[1:]:
            value_int_map = bidict(enumerate(pd.unique(self.__dataset[col])))
            value_int_map_str = bidict(enumerate(list(map(str,pd.unique(self.__dataset[col])))))
            self.column_value_dict[col] = value_int_map
            self.column_value_dict_str[col] = value_int_map_str
        # transform dataset to integer values
        for col, value_int_map in self.column_value_dict.items():
            self.__dataset[col] = self.__dataset[col].map(value_int_map.inverse)

        self.time_stamps = self.__dataset.iloc[:,0]
        self.flow_features = self.__dataset.iloc[:,1:]
        self.shape = self.flow_features.shape
        self.col_name_map = bidict(enumerate(self.flow_features.columns))
        self.flow_features.rename(columns=self.col_name_map.inverse, inplace=True)

        #frequency of each value in each column
        self.value_prob = {}
        for col in self.flow_features.columns:
            self.value_prob[col] = self.flow_features[col].value_counts(normalize=True)

        self.never_fix = set([self.col_name_map.inverse[x] for x in domain_knowledge.never_fix])
        self.placeholder_set_columns = set([self.col_name_map.inverse[x] for x in domain_knowledge.placeholder_set_columns])
        self.placeholder_get_columns = set([self.col_name_map.inverse[x] for x in domain_knowledge.placeholder_get_columns])
        self.ip_columns = set([self.col_name_map.inverse[x] for x in domain_knowledge.ip_columns])
        self.cont_repr = None
        self.time_precition=100
    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load_model(filename) -> 'Dataset':
        return pickle.load(open(filename, 'rb'))


def reconstruct_bytes(col, dic, v2=True):
    if not v2:
        ser = col.astype(float)
        name = col.name
        n = dic[name]["n_components"]
        #n = len(dic[name]["weights"])
        mean_array = np.array(dic[name]["means"])
        std_array = np.array(dic[name]["std"])
        edge = dic[name]["edge"]
        M = dic[name]["max"]
        ser.loc[ser == float(n)] = np.random.uniform(edge, M, (ser == float(n)).sum())
        norm = ser.loc[ser < float(n)]
        mean = np.array(mean_array)[norm.values.astype(int)]
        if np.array(std_array).shape != (): #Take into accont the case where the covariance of the GMM is "tied"
            std = np.array(std_array)[norm.values.astype(int)]
        else:
            std = float(std_array)
        ser.loc[ser < float(n)] = np.random.normal(mean, np.sqrt(std))
        ser=np.maximum(ser, np.zeros(len(ser)))
        return ser
    else:
        ser = col.astype(int)
        name = col.name
        n = len(dic[name])
        temp = ser[ser!=0]
        temp.replace(dic[name], inplace=True)
        temp = temp.apply(lambda x: np.random.uniform(low=x.left, high=x.right))
        ser.loc[ser!=0] = temp
        return ser

def load_CIDDS_dataset(path):

    df = pd.read_csv(path)
    train = preprocess_CIDDS_dataset(df)
    train_d, discrete_dic = discretize(train)

    dataset = Dataset(train_d.copy())
    dataset.cont_repr = get_CIDDS_cont_repr(train, discrete_dic)

    return dataset

def load_CIDDS_splitted_dataset(path, n_split, split_before=False):
    df = pd.read_csv(path)
    train = preprocess_CIDDS_dataset(df)
    train_d, discrete_dic = discretize(train, transform_timestamps = False)
    #We need to transform the timestamps separately from the rest so the continuous representation will keep the initial timestamps
    cont_repr = get_CIDDS_cont_repr(train_d, discrete_dic)
    train_d['Date first seen'] = time_to_int(train_d['Date first seen'], 100)
    global_dataset = Dataset(train_d.copy())
    global_dataset.cont_repr = cont_repr
    if split_before:
        #TODO Split Before doesn't work still raise the KeyError:454
        chunks = np.asarray([deepcopy(global_dataset) for i in range(n_split)])
        split_idx = np.array_split(global_dataset.flow_features.index, n_split)
        for i in range(len(chunks)):
            chunks[i].flow_features = global_dataset.flow_features.iloc[split_idx[i],:].reset_index(drop=True)
            chunks[i].time_stamps = global_dataset.time_stamps.iloc[split_idx[i]].reset_index(drop=True)
    else:
        chunks = np.array_split(train_d, n_split)
        for i in range(len(chunks)):
            dataset = Dataset(chunks[i].reset_index(drop=True).copy())
            dataset.cont_repr = get_CIDDS_cont_repr(chunks[i], discrete_dic)
            dataset.cont_repr.add_first_flow_time(cont_repr.get_first_flow_time())
            chunks[i] = dataset

    return global_dataset, chunks

#TODO define fix data format to return

def load_train_set(sample_size=None):
    df = pd.read_csv('data/train.csv')
    df= df.head(sample_size) if sample_size is not None else df

    df.sort_values(by=['Day', 'Time'], inplace=True)

    return df


# FIXME remove

def load_cicids17(sample_size=None):
    df = pd.read_csv('data/CIDDS_001_train.csv')
    df= df.head(sample_size) if sample_size is not None else df

    # cast DstPort to int where Proto is ICMP
    df.loc[df['Proto'] == 'ICMP', 'DstPort'] = df.loc[df['Proto'] == 'ICMP', 'DstPort'].astype(int)

    return df

# FIXME remove

def preprocess_socbed_bi(data, first_attack_ts, end_ts=None, sample_size=None, precition=100):
    #TODO return necessary information to transform data back, so discretization can be reversed
    df= data.head(sample_size) if sample_size is not None else data

    df['Date first seen'] = pd.to_datetime(df['Date first seen'])
    # drop all rows before first "attack"
    df = df[df['Date first seen'] > first_attack_ts]
    if end_ts is not None:
        df = df[df['Date first seen'] < end_ts]

    df['Proto'] = df['Proto'].str.strip()
    df['Src IP Addr'] = df['Src IP Addr'].str.strip()
    df['Dst IP Addr'] = df['Dst IP Addr'].str.strip()

    df["External"] = df["Dst IP Addr"].apply(lambda x: not x.startswith("172") )

    df['In Byte'] = bytes_to_int(df['In Byte'])
    df['Out Byte'] = bytes_to_int(df['Out Byte'])

    df.reset_index(inplace=True, drop=True)

    return df

# FIXME remove

def preprocess_IoT_dataset(data, first_attack_ts, end_ts=None, sample_size=None, precition=100):
    #TODO return necessary information to transform data back, so discretization can be reversed
    df= data.head(sample_size) if sample_size is not None else data

    df['Date first seen'] = pd.to_datetime(df['Date first seen'])
    # drop all rows before first "attack"
    df = df[df['Date first seen'] > first_attack_ts]
    if end_ts is not None:
        df = df[df['Date first seen'] < end_ts]

    df["From Ext."] = df["Dst IP Addr"].apply(lambda x: not x.startswith("192.168") )
    df["To Multicast"] = df["Dst IP Addr"].apply(lambda x: x.startswith("ff0") )

    df = df.sort_values("Date first seen")
    df.reset_index(inplace=True, drop=True)

    return df

def parse_tcp_flags(tcp_flag_str):
    # Define the possible flags and their meanings


    # Initialize a dictionary to store the flags status
    parsed_flags = {flag: False for flag in flag_map.values()}

    # Iterate over each character in the input string
    for char in tcp_flag_str:
        if char in flag_map:
            parsed_flags[flag_map[char]] = True

    return parsed_flags


def preprocess_CIDDS_dataset(data):
    df= data.copy()

    df['Date first seen'] = pd.to_datetime(df['Date first seen'])

    df['Proto'] = df['Proto'].str.strip()
    df['Src IP Addr'] = df['Src IP Addr'].str.strip()
    df['Dst IP Addr'] = df['Dst IP Addr'].str.strip()
    df['Flags'] = df['Flags'].str.strip()


    # for i, row in df.iterrows():
    #     for f,v in parse_tcp_flags(row['Flags'].strip()).items():
    #         df.at[i,f] = v



    # df.drop(columns=['Flags'], inplace=True)

    df['Dst Pt'] = df['Dst Pt'].astype(float)
    df['Duration'] = df['Duration'].astype(float)

    df['In Byte'] = df['In Byte'].astype(int)
    df['In Packet'] = df['In Packet'].astype(int)
    df['Out Byte'] = df['Out Byte'].astype(int)
    df['Out Packet'] = df['Out Packet'].astype(int)

    #df["From Ext."] = df["Src IP Addr"].str.contains("_")

    df = df.sort_values("Date first seen")
    df.reset_index(inplace=True, drop=True)

    return df

def discretize(df, n_components_range=[1,100], optimize=False, transform_timestamps=True, v2=True):

    data = df.copy()

    continuous = data.loc[:,["In Byte", "Out Byte", "In Packet", "Out Packet", "Duration"]]
    dic = {}
    for col in continuous.columns:
        if not v2:
            m = continuous[col].describe([.9])[5]
            temp = continuous.loc[data[col]<=m, col].to_numpy().reshape(-1,1)
            if optimize:
                best_model, best_params = optimize_gmm(temp, n_components_range)
                print(f"For {col}, the best parameters of the GMM are : ", best_params)
                best_n_components = best_params["n_components"]
            else :
                if col == "In Byte":
                    best_n_components = 42
                    best_model = GaussianMixture(n_components=best_n_components, covariance_type='diag', random_state=42).fit(temp)
                if col == "Out Byte":
                    best_n_components = 43
                    best_model = GaussianMixture(n_components=best_n_components, covariance_type='diag', random_state=42).fit(temp)
                if col == "In Packet":
                    best_n_components = 41
                    best_model = GaussianMixture(n_components=best_n_components, covariance_type='tied', random_state=42).fit(temp)
                if col == "Out Packet":
                    best_n_components = 40
                    best_model = GaussianMixture(n_components=best_n_components, covariance_type='tied', random_state=42).fit(temp)
                if col == "Duration":
                    best_n_components = 40
                    best_model = GaussianMixture(n_components=best_n_components, covariance_type='spherical', random_state=42).fit(temp)
            dic[col] = {"edge":m, "max": continuous[col].max(), "weights": best_model.weights_.squeeze(), "n_components": best_n_components, "means": best_model.means_.squeeze(), "std": best_model.covariances_.squeeze()}
            continuous.loc[data[col]<=m, col] = best_model.predict(temp)
            continuous.loc[data[col]>m, col] = best_n_components
            data[col] = continuous[col]
        else:
            c = continuous[col]
            j = c.copy()
            c = c[c!=0]
            c = pd.qcut(c, 40, duplicates="drop")
            j.loc[j!=0] = c
            d = bidict(zip(list(range(1,c.nunique()+1)), c.cat.categories))
            d[0] = 0
            j = j.replace(d.inverse).astype(int)
            data[col] = j
            dic[col] = d
    if transform_timestamps:
        data['Date first seen'] = time_to_int(df['Date first seen'], 100)

    return data, dic



def get_CIDDS_cont_repr(data, dic, precition=100) -> ContinousRepr:
    cont_rept = ContinousRepr()
#    df = load_socbed_bi(data, precition=precition, original_values=True, eva=eva)
    df = data.copy()
    cont_rept.add_first_flow_time(df['Date first seen'].iloc[0])
    cont_rept.add_time_precision(precition)
    # in bytes cut points
    cont_rept.add_cutpoints(dic)
    return cont_rept


