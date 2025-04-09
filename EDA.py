from sklearn.datasets import load_wine 
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
data = load_wine()


df_data = pd.DataFrame(data=data["data"], columns = data["feature_names"])
pd.set_option('display.max_columns', None)


print(df_data.head())

stats = df_data.describe()
print(stats)

df_data["target"] = data["target"]

plotted_features = df_data.columns[:9] 
seaborn.pairplot(df_data, vars=plotted_features, hue="target", palette="Set1", diag_kind="kde")
plt.show()
