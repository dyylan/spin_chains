import pandas as pd


def update_data(protocol, chain, alpha, save_tag, data_dict, replace=True):
    filepath = f"data/{protocol}_protocol/{chain}_chain/alpha={alpha}/{save_tag}.csv"
    try:
        df = pd.read_csv(filepath, index_col=False)
        new_df = pd.DataFrame(data_dict)
        df = df.append(new_df)
        if replace:
            df = df.drop_duplicates(subset="spins", keep="last")
            # df.spins = df.spins.astype(float)
            df = df.sort_values(by=["spins"])
        else:
            df = df.drop_duplicates(subset="spins", keep="first")
    except FileNotFoundError:
        df = pd.DataFrame(data=data_dict)
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
