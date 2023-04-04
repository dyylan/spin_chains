import pandas as pd
import numpy as np


def update_data(
    protocol,
    chain,
    alpha,
    save_tag,
    data_dict,
    replace=False,
    replace_col="spins",
    keep="last",
):
    filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{save_tag}.csv"
    if replace:
        try:
            df = pd.read_csv(filepath, index_col=False)
            new_df = pd.DataFrame(data_dict)
            df = df.append(new_df)
            if replace:
                df = df.drop_duplicates(subset=replace_col, keep="last")
                # df.spins = df.spins.astype(float)
                df = df.sort_values(by=[replace_col])
            else:
                df = df.drop_duplicates(subset=replace_col, keep="first")
        except FileNotFoundError:
            df = pd.DataFrame(data=data_dict)
    else:
        df = pd.DataFrame(data_dict)
    df.to_csv(filepath, index=False)
    return df.to_dict()


def read_data(protocol, chain, alpha, save_tag):
    filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{save_tag}.csv"
    df = pd.read_csv(filepath, index_col=False)
    return df.to_dict("list")


def read_data_spin(protocol, chain, alpha, save_tag, spins):
    filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{save_tag}.csv"
    df = pd.read_csv(filepath, index_col="spins")
    df_row = df.loc[spins]
    print(df_row)
    print(df_row.to_dict())
    return df_row.to_dict()


def update_state_data(
    protocol,
    chain,
    alpha,
    n_ions,
    mu,
    z_trap_frequency,
    data_dict,
    data_keys,
    r=False,
    replace=True,
):
    for key in data_keys:
        r_tag = f"_r={r}" if r else ""
        filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{key}_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}{r_tag}.csv"
        data = {"time_microseconds": int(data_dict["time_microseconds"])}
        data.update({f"{i + 1}": v[0] for i, v in enumerate(data_dict[key])})
        try:
            df = pd.read_csv(filepath, index_col="time_microseconds")
            new_df = pd.DataFrame.from_records([data], index="time_microseconds")
            df = df.append(new_df)
            if replace:
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
            else:
                df = df[~df.index.duplicated(keep="first")]
        except FileNotFoundError:
            df = pd.DataFrame.from_records([data], index="time_microseconds")
        df.to_csv(filepath, index="time_microseconds")
    return df.to_dict()


def read_state_data(
    protocol, chain, alpha, n_ions, mu, z_trap_frequency, data_key, time, r=False
):
    r_tag = f"_r={r}" if r else ""
    filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{data_key}_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}{r_tag}.csv"
    df = pd.read_csv(filepath, index_col="time_microseconds")
    df_row = df.loc[time]
    d = np.complex128(list(df_row))
    return d


def read_all_state_data(
    protocol, chain, alpha, n_ions, mu, z_trap_frequency, data_key, r=False
):
    r_tag = f"_r={r}" if r else ""
    filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{data_key}_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}{r_tag}.csv"
    df = pd.read_csv(filepath, index_col="time_microseconds")
    d = df.to_dict("list")
    states = [[complex(d[k][i]) for k in d] for i in range(len(df.index))]
    data = {"time_microseconds": list(df.index), "states": states}
    return data
