import pandas as pd
from matplotlib import scale
from sklearn import datasets
import numpy as np
import matplotlib as plt

# attempt at scaling the features of the pd speech features csv file by identifying the colums and using scale.fit_transform
print("Question 2------------------- : ")
df = pd.read_csv("pd_speech_features.csv")
X = df[
    'id', 'gender', 'PPE', 'DFA', 'RPDE', 'numPulses', 'numPeriodsPulses', 'meanPeriodPulses', 'stdDevPeriodPulses', 'locPctJitter'
    , 'locAbsJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'meanAutoCorrHarmonicity '
    , 'meanNoiseToHarmHarmonicity', 'meanHarmToNoiseHarmonicity', 'minIntensity', 'maxIntensity', 'meanIntensity', 'f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4', 'GQ_prc5_95', 'GQ_std_cycle_open'
    , 'GQ_std_cycle_closed', 'GNE_mean', 'GNE_std', 'GNE_SNR_TKEO', 'GNE_SNR_SEO', 'GNE_NSR_TKEO', 'GNE_NSR_SEO', 'VFER_mean', 'VFER_std', 'VFER_entropy', 'VFER_SNR_TKEO', 'VFER_SNR_SEO', 'VFER_NSR_TKEO'
    , 'VFER_NSR_SEO', 'IMF_SNR_SEO', 'IMF_SNR_TKEO', 'IMF_SNR_entropy', 'IMF_NSR_SEO', 'IMF_NSR_TKEO', 'IMF_NSR_entropy', 'mean_Log_energy', 'mean_MFCC_0th_coef', 'mean_MFCC_1st_coef', 'mean_MFCC_2nd_coef'
    , 'mean_MFCC_3rd_coef', 'mean_MFCC_4th_coef', 'mean_MFCC_5th_coef', 'mean_MFCC_6th_coef', 'mean_MFCC_7th_coef', 'mean_MFCC_8th_coef', 'mean_MFCC_9th_coef', 'mean_MFCC_10th_coef', 'mean_MFCC_11th_coef'
    , 'mean_MFCC_12th_coef', 'mean_delta_log_energy', 'mean_0th_delta', 'mean_1st_delta', 'mean_2nd_delta', 'mean_3rd_delta', 'mean_4th_delta', 'mean_5th_delta', 'mean_6th_delta', 'mean_7th_delta', 'mean_8th_delta'
    , 'mean_9th_delta', 'mean_10th_delta', 'mean_11th_delta', 'mean_12th_delta', 'mean_delta_delta_log_energy', 'mean_delta_delta_0th', 'mean_1st_delta_delta', 'mean_2nd_delta_delta', 'mean_3rd_delta_delta', 'mean_4th_delta_delta'
    , 'mean_5th_delta_delta', 'mean_6th_delta_delta', 'mean_7th_delta_delta', 'mean_8th_delta_delta', 'mean_9th_delta_delta', 'mean_10th_delta_delta', 'mean_11th_delta_delta', 'mean_12th_delta_delta', 'std_Log_energy'
    , 'std_MFCC_0th_coef', 'std_MFCC_1st_coef', 'std_MFCC_2nd_coef', 'std_MFCC_3rd_coef', 'std_MFCC_4th_coef', 'std_MFCC_5th_coef', 'std_MFCC_6th_coef', 'std_MFCC_7th_coef', 'std_MFCC_8th_coef', 'std_MFCC_9th_coef', 'std_MFCC_10th_coef'
    , 'std_MFCC_11th_coef', 'std_MFCC_12th_coef', 'std_delta_log_energy', 'std_0th_delta', 'std_1st_delta', 'std_2nd_delta', 'std_3rd_delta', 'std_4th_delta', 'std_5th_delta', 'std_6th_delta', 'std_7th_delta', 'std_8th_delta'
    , 'std_9th_delta', 'std_10th_delta', 'std_11th_delta', 'std_12th_delta', 'std_delta_delta_log_energy', 'std_delta_delta_0th', 'std_1st_delta_delta', 'std_2nd_delta_delta', 'std_3rd_delta_delta', 'std_4th_delta_delta'
    , 'std_5th_delta_delta', 'std_6th_delta_delta', 'std_7th_delta_delta', 'std_8th_delta_delta', 'std_9th_delta_delta', 'std_10th_delta_delta', 'std_11th_delta_delta', 'std_12th_delta_delta', 'Ea', 'Ed_1_coef', 'Ed_2_coef', 'Ed_3_coef'
]
scaledX = scale.fit_transform(X)
print(scaledX)

# attempt at SVM using the iris database since figuring it out with the other one was kind of confusing
print("part 2 - SVM-------------------")
print("with iris dataset since could not figure out the other")
# loading dataset in and identifying the targets
iris = datasets.load_iris()
iris.keys()
iris = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

iris.head()
species = []

# for loop to identify the targets accordingly
for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species

iris.head()
setosa = iris[iris.species == "setosa"]
versicolor = iris[iris.species=='versicolor']
virginica = iris[iris.species=='virginica']

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # adjusting the length and width of plot

# labels and scatter points
ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label="Setosa Petal", facecolor="blue")
ax.scatter(versicolor['petal length (cm)'], versicolor['petal width (cm)'], label="Versicolor", facecolor="green")
ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'], label="Virginica", facecolor="red")

ax.set_xlabel("sepal length (cm)")
ax.set_ylabel("sepal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()

