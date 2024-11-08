import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def parse_byte_annotation(annot):
    assert(isinstance(annot,str))
    anno = annot.strip()
    multipliers = {'k': 1e3, 'm': 1e6, 'g': 1e9, 't': 1e12}
    if annot[-1].lower() in multipliers:
        value = float(annot[:-1])
        multiplier = multipliers[annot[-1].lower()]
        return value * multiplier
    else:
        return float(anno)

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def optimize_gmm(X, n_components_range):
    param_grid = {
    "n_components": range(n_components_range[0], n_components_range[1]+1),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
    grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score, verbose=4, n_jobs=-1
)
    grid_search.fit(X)
    return grid_search.best_estimator_, grid_search.best_params_

def bytes_to_int(ser):
    assert(isinstance(ser,pd.Series))
    return ser.apply(parse_byte_annotation)

def time_to_int(ser, precition=100):
    assert(isinstance(ser,pd.Series))
    assert(ser.dtype == 'datetime64[ns]')
    # to unix timestamp
    ser = ser.apply(lambda x: round(x.timestamp()*precition))
    #substract first date from all dates
    ser = ser - ser.iloc[0]
    return ser

def split_comm(comm, time_of_conn_seconds):
    df = comm.copy()
    df["Date first seen"] = pd.to_datetime(df["Date first seen"])
    df = df.sort_values("Date first seen").reset_index()
    df.drop("index", axis=1, inplace=True)
    time_diffs = pd.to_datetime(df["Date first seen"]).diff(periods = -1).dt.total_seconds()+df["Duration"].astype(float)
    flags_idxs = df[df['Flags'].str.contains('F')].index
    flags_idxs = list(flags_idxs[1::2])
    time_idxs = list(time_diffs[time_diffs.abs() >= time_of_conn_seconds].index)
#    split_indices = sorted(set([df.index[0]] + flags_idxs + time_idxs + [df.index[-1]])) #We do not take int account the Flags
    split_indices = sorted(set([df.index[0]] + time_idxs + [df.index[-1]]))
    sessions = []
    for i in range(1, len(split_indices)):
        if i ==1:
            sessions.append(df.loc[:split_indices[1]])
        else :
            sessions.append(df.loc[split_indices[i-1]+1:split_indices[i]])
    return sessions

def convert_unidirectionnal_into_bidirectionnal(path = "/media/aschoen/KINGSTON/dataset/WISENT-CIDDS/WISENT-CIDDS-001/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week3.csv", time_of_conn_seconds = 120):
    df = pd.read_csv(path)
    df = df[df["class"]=="normal"]
    df1 = df[["Date first seen", "Proto", 'Src IP Addr', "Dst IP Addr", "Src Pt", "Dst Pt", "Bytes", "Packets", "Flags", "Duration"]]

    df1.loc[:,"Bytes"] = bytes_to_int(df1["Bytes"].astype(str)) # Replace "1M" per e6
    df1.loc[:,"Src Pt"] = df1["Src Pt"].astype(float) #Should be in the same format as Dst Pt

    temp = df1.astype(str)
    temp["key"] = temp[["Proto", 'Src IP Addr', "Dst IP Addr", "Src Pt", "Dst Pt"]].apply(sorted, axis=1).apply(tuple) #Create the groupby key, so that we have one datafram per communication. Better to do this than multiple groupby so we will not considere dupplicate.
    # temp["key"] = list(zip(
    #     temp["Proto"],
    #     temp['Src IP Addr'],
    #     temp["Dst IP Addr"],
    #     temp["Src Pt"],
    #     temp["Dst Pt"]
    #     )) #Create the groupby key, so that we have one datafram per communication. Better to do this than multiple groupby so we will not considere dupplicate.

    result = []

    grouped = temp.groupby(temp["key"]) #Do the group by

    with tqdm(total = len(grouped), desc="Processing") as pbar:
        for key, comm in grouped:
            comm.drop("key",axis=1,inplace=True) #We do no longer need the key
            sessions = split_comm(comm, time_of_conn_seconds) #We splitt the dataframe into multiple chunk, each of this corresponding to one session
            for sess in sessions: #For each sessions
                session = sess.copy() #To avoid alert from Pandas
                if comm["Src IP Addr"].nunique()>1: #The case of two side communication like all TCP or DNS
                    Out_Byte = sess.loc[sess["Src IP Addr"] == sess["Src IP Addr"].iloc[0],"Bytes"].astype(float).sum()
                    Out_Packet = sess.loc[sess["Src IP Addr"] == sess["Src IP Addr"].iloc[0],"Packets"].astype(float).sum()
                    In_Byte = sess.loc[sess["Src IP Addr"] == sess["Dst IP Addr"].iloc[0],"Bytes"].astype(float).sum()
                    In_Packet = sess.loc[sess["Src IP Addr"] == sess["Dst IP Addr"].iloc[0],"Packets"].astype(float).sum()
                else: #If there is only one sender, then In_ Bytes should be 0, like ICMP etc..
                    Out_Byte = sess["Bytes"].astype(float).sum()
                    Out_Packet = sess["Packets"].astype(float).sum()
                    In_Byte = 0
                    In_Packet = 0

                #DataFrame where each column represents positions in the flag strings
                flags_sess = session["Flags"].str.extractall('(.)')[0].unstack(fill_value='.')
                #check for the presence of flags other than '.' in each position and then aggregate into a flag string
                aggregated_flags = ''.join(flags_sess.apply(lambda col: col[col != '.'].unique()[0] if col[col != '.'].any() else '.', axis=0))

                session_end = max(session["Date first seen"]+pd.to_timedelta(session["Duration"].astype(float), unit='s'))
                session_dur = session_end - min(session["Date first seen"])
                session_dur = session_dur.total_seconds()

                session.drop(["Bytes", "Packets", "Flags", "Duration"], axis=1, inplace=True) #We no longer need Bytes
                session = session.iloc[0] #Just consider the first flow as a summary
                session['In Byte'] = In_Byte
                session['In Packet'] = In_Packet
                session['Out Byte'] = Out_Byte
                session['Out Packet'] = Out_Packet
                session["Flags"] = aggregated_flags
                session["Duration"] = session_dur

                if float(session["Src Pt"]) < float(session["Dst Pt"]): #If the destination port is inferior to the source port
                    session.rename({"Src IP Addr":"Dst IP Addr","Dst IP Addr":"Src IP Addr", "Src Pt":"Dst Pt", "Dst Pt":"Src Pt", "In Byte":"Out Byte", "Out Byte":"In Byte", "In Packet":"Out Packet", "Out Packet":"In Packet"}, inplace = True) #Reverse the diection of the flow
                result.append(pd.DataFrame(session).transpose())
                pbar.update(1)

    bi_dic_df = pd.concat(result) #concatenate all the session
    bi_dic_df.drop("Src Pt", axis=1, inplace=True)
    bi_dic_df = bi_dic_df.sort_values("Date first seen")
    bi_dic_df.reset_index(inplace=True, drop=True)
    return bi_dic_df


def preprocess_test_for_patterns(df):
    assert(isinstance(df,pd.DataFrame))
    assert(list(df.columns) == ['Day', 'Time', 'Duration', 'Proto', 'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 'Packets', 'Bytes', 'Flags', 'class', 'attackType'])

    df.drop(['Day', 'Time', 'Duration','Flags', 'class', 'attackType'], axis=1, inplace=True)
    # ignore bytes for now as numeric value, in practice replace by simple prediction model
    df.drop(['Bytes'], axis=1, inplace=True)

    df = df.astype({'Src Pt': 'object', 'Dst Pt': 'object', 'Packets': 'object'})

    return df