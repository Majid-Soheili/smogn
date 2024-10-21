# 01. Importing Libraries =====================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from smogn.SkewedSmoter import SkewedSmoter

# 02. Load Data ==============================================
path = "/Volumes/Researches/CHS/CLIMOS/SFFA/resources/data/processed/v3/training/trainingWS.csv"
data = pd.read_csv(path)

target = "dst"
fss = ["spec_code", "aspect", "slope", "sand", "clay", "elevation", "swc6_14", "tmax_365", "lc",
       "pet_49", "pre_7", "swc6_1", "swc4_365", "tmax_330", "pet_90", "tavg_365", "tmin_1", "tmin_63",
       "tmin_56", "swc1_330"]

data = data[fss + [target]]
data[target] = np.log(data[target] + 1)

# filter the data such that it just contains the spec_code 1
data = data[data['spec_code'] == 1]
data['spec_code'] = data['spec_code'].astype('category')
data['lc'] = data['lc'].astype('category')
data.reset_index(drop=True, inplace=True)

# 03. Data Preprocessing ======================================

balancer = SkewedSmoter(data=data, target=target)
balancer.set_focus(3)
balancer.set_skewness(0.3)
balancer.set_steepness(1.5)
synthetic_data = balancer.generate_synthetic_data()

print(data.shape[0])
print(synthetic_data.shape[0])

sns.kdeplot(data[target], label="Original")
sns.kdeplot(synthetic_data[target], label="Modified")
#sns.histplot(synthetic_data[target], bins=17, kde=False, label="Original", alpha=0.2)


# Add title and labels
plt.title('KDE Plot of dst')
plt.xlabel('dst')
plt.ylabel('Density')

# Show legend
plt.legend()

# Display the plot
plt.show()


print("Finished")