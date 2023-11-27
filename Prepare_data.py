import numpy as np
import pandas as pd

class Prepare:
    def __init__(self,data):
        self.df : pd.DataFrame = data
        self.raw_df = data

    def convert_M(self,x:str):
        if x.endswith("M"):
            x = x.replace("M", ' ')
            x = float(x) * 1000000
            return x
        elif x.endswith("K"):
            x = x.replace("K", ' ')
            x = float(x) * 1000
            return x
        elif x.endswith("B"):
            x = x.replace("B", ' ')
            x = float(x) * 1000000000
            return x
        else:
            return x

    def replace_comma(self):
        self.df["Price"] = self.df["Price"].str.replace(',', '')
        self.df["Open"] = self.df["Open"].str.replace(',', '')
        self.df["High"] = self.df["High"].str.replace(',', '')
        self.df["Low"] = self.df["Low"].str.replace(',', '')
        vol = []
        for i in self.df["Vol."]:
            vol.append(self.convert_M(i))
        self.df["Vol."] = vol

    def to_list2D(self):
        d = []
        for i in self.df:
            d.append(i[0])
        return d

