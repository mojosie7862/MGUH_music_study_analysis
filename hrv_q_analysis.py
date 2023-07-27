import pandas as pd
from scipy.stats.stats import pearsonr, ttest_ind
import numpy as np
import matplotlib.pyplot as plt

qs_data = pd.read_csv("NHC_q_data_cohort1.csv")
cortisol_data = pd.read_csv('all_cortisol_data.csv')

qs_data['delt_cortisol'] = qs_data['cortisol_2_mean'] - qs_data['cortisol_1_mean']
qs_data['delt_STAI_state'] = qs_data['STAI_2 State'] - qs_data['STAI_1 State']
qs_data['delt_STAI_trait'] = qs_data['STAI_2 Trait'] - qs_data['STAI_1 State']

cols = ['STAI_1 State', 'STAI_1 Trait', 'delt_STAI_state', 'delt_STAI_trait', 'Emotional sensitivity to music', 'Personal commitment to music',
 'Music memory and imagery', 'Listening sophistication', 'Indifference to music', 'Musical transcendance', 'Emotion regulation',
 'Social', 'Music identity and expression', 'Cognitive regulation', 'head_circ', 'head_ni', 'delt_cortisol', 'cortisol_1_mean', 'cortisol_2_mean']
qs_data = qs_data[cols]

corr_matirx = qs_data.corr()
# corr_matirx.to_csv('qs_corr_matrix.csv')

v = pearsonr(qs_data['delt_STAI_state'], qs_data['delt_STAI_trait'])

cort_b1 = cortisol_data['cortisol_1_mean'].values
cort_b2 = cortisol_data['cortisol_2_mean'].values
state_b1 = cortisol_data['STAI_1 State'].values
state_b2 = cortisol_data['STAI_2 State'].values

cort_df = pd.DataFrame({'After': cort_b2, 'Before': cort_b1})

cort_tt = ttest_ind(a=cort_b1, b=cort_b2, equal_var=True)
print('n =', len(cort_df))
print('cortisol', cort_tt)
stai_tt = ttest_ind(a=state_b1, b=state_b2, equal_var=True)
print('stai', stai_tt)

# x = np.arange(len(cort_df))
# y1 = cort_df['before']
# y2 = cort_df['after']
# width = 0.4
# plt.bar(x-0.2, y1, width)
# plt.bar(x+0.2, y2, width)
plt.boxplot(cort_df, vert=False, showfliers=False, showmeans=True)
plt.yticks(ticks=[1,2], labels=cort_df.columns)
plt.xlabel('Cortisol (ug/dL)')
#plt.title("Change in Salivary Cortisol Through Music Presentation")
plt.show()


# for x in qs_data.columns:
#  for y in qs_data.columns:
#   if y != x:
#    i = list(qs_data[x])
#    j = list(qs_data[y])
#    corr_res = pearsonr(list(qs_data[x]), list(qs_data[y]))
#    if corr_res[1] < 0.05:
#     print(x)
#     print(y)
#     print(corr_res)