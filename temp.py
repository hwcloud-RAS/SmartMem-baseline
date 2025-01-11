import feather
import pandas as pd

csv = pd.read_csv(r"E:\Wechat Files\WeChat Files\wxid_vt54i975je3522\FileStorage\File\2025-01\sn_2.csv")

df = feather.read_dataframe(r"E:\Wechat Files\WeChat Files\wxid_vt54i975je3522\FileStorage\File\2025-01\sn_2.feather")

print(df)
