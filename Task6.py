import pandas as pd
import os
import numpy as np
import re
from cache_em_all import Cachable
DATA_DIR = "data/training_setA/" # Path to the data
# Names of all columns in the data that contain physiological data
physiological_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
'Fibrinogen', 'Platelets']
# Names of all columns in the data that contain demographic data
demographic_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
# The combination of physiological and demographic data is what we will use as features in
feature_cols = physiological_cols + demographic_cols
# The name of the column that contains the value we are trying to predic
label_col = "SepsisLabel"
# Pre-calculated means and standard deviation of all physiological and demographic columns.
# data using their z-score. This isn't as important for simpler models such as random fores # but can result in significant improvements when using neural networks
physiological_mean = np.array([
83.8996, 97.0520, 36.8055, 126.2240, 86.2907,
66.2070, 18.7280, 33.7373, -3.1923, 22.5352,
0.4597, 7.3889, 39.5049, 96.8883, 103.4265,
22.4952, 87.5214, 7.7210, 106.1982, 1.5961,
0.6943, 131.5327, 2.0262, 2.0509, 3.5130,
4.0541, 1.3423, 5.2734, 32.1134, 10.5383,
38.9974, 10.5585, 286.5404, 198.6777])
physiological_std = np.array([
demographic_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
demographic_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])
1
17.6494, 3.0163, 0.6895, 24.2988, 16.6459,
14.0771, 4.7035, 11.0158, 3.7845, 3.1567,
6.2684, 0.0710, 9.1087, 3.3971, 430.3638,
19.0690, 81.7152, 2.3992, 4.9761, 2.0648,
1.9926, 45.4816, 1.6008, 0.3793, 1.3092,
0.5844, 2.5511, 20.4142, 6.4362, 2.2302,
29.8928, 7.0606, 137.3886, 96.8997])
def load_single_file(file_path):
df = pd.read_csv(file_path, sep='|')
df['hours'] = df.index
df['patient'] = re.search('p(.*?).psv', file_path).group(1)
return df
def clean_data(data):
data.reset_index(inplace=True, drop=True)
# Normalizes physiological and demographic data using z-score.
data[physiological_cols] = (data[physiological_cols] - physiological_mean) / physiologi
data[demographic_cols] = (data[demographic_cols] - demographic_mean) / demographic_std
# Maps invalid numbers (NaN, inf, -inf) to numbers (0, really large number, really smal
data[feature_cols] = np.nan_to_num(data[feature_cols])
return data
def get_data_files():
return [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR)) if int(x[1:-4])
@Cachable('training_setA.csv')
def load_data():
data = get_data_files()
data_frames = [clean_data(load_single_file(d)) for d in data]
merged = pd.concat(data_frames)
return merged
paths = get_data_files()
for p in paths[:10]:
print(re.search('p(.*?).psv', p).group(1))
000001
000002
000003
000004
000006
000007
000008
000009
000011
000012
df = load_data()
2
df.isnull().any()
HR False
O2Sat False
Temp False
SBP False
MAP False
DBP False
Resp False
EtCO2 False
BaseExcess False
HCO3 False
FiO2 False
pH False
PaCO2 False
SaO2 False
AST False
BUN False
Alkalinephos False
Calcium False
Chloride False
Creatinine False
Bilirubin_direct False
Glucose False
Lactate False
Magnesium False
Phosphate False
Potassium False
Bilirubin_total False
TroponinI False
Hct False
Hgb False
PTT False
WBC False
Fibrinogen False
Platelets False
Age False
Gender False
Unit1 False
Unit2 False
HospAdmTime False
ICULOS False
SepsisLabel False
hours False
patient False
dtype: bool
3
4
# Filter tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tens
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)
import re
import altair as alt
from stable_baselines.deepq import DQN, MlpPolicy as DQN_MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from env.SepsisEnv import SepsisEnv
from load_data import load_data
from add_reward import add_reward_df, add_end_episode_df
df = load_data()
df = add_reward_df(df)
df = add_end_episode_df(df)
df = df.reset_index()
total_timesteps = 10_000
iterations = 50_000
def train_model(env, model, total_timesteps, iterations):
model.learn(total_timesteps=total_timesteps)
reward_list = []
obs = env.reset()
patient_count = 0
for _ in tqdm(range(iterations)):
action, _states = model.predict(obs)
obs, rewards, done, info = env.step(action)
reward_list.append(rewards[0])
if done:
1
patient_count += 1
obs = env.reset()
model_name = re.sub(r'\W+', '', str(model. class ).split('.')[-1])
policy_name = re.sub(r'\W+', '', str(model.policy).split('.')[-1])
# print('Model: ', model_name)
# print('Policy: ', policy_name)
# print('Total patients: ', patient_count)
# print('Total reward:', sum(reward_list))
return sum(reward_list)
Bayesian Optimization code from: https://colab.research.google.com/gist
/iyaja/bf1d35a09ea5e0559900cc9136f96e36/hyperparameter-optimizationfastai.ipynb#scrollTo=gGZm73Txs9PS
def fit_with(lr, bs, eps, final_eps):
env = DummyVecEnv([lambda: SepsisEnv(df)])
model = DQN(env=env,
policy=DQN_MlpPolicy,
learning_rate=lr,
buffer_size=bs,
exploration_fraction=eps,
exploration_final_eps=final_eps,
)
total_reward = train_model(env=env, model=model, total_timesteps=total_timesteps, itera
return total_reward
# Bounded region of parameter space
pbounds = {'lr': (1e-2, 1e-4), 'bs':(5_000, 100_000), 'eps':(0.01, 0.2), 'final_eps': (0.01
optimizer = BayesianOptimization(
f=fit_with,
pbounds=pbounds,
verbose=2
)
optimizer.maximize(init_points=2, n_iter=5,)
for i, res in enumerate(optimizer.res):
print("Iteration {}: \n\t{}".format(i, res))
print('Max', optimizer.max)
| iter | target | bs | eps | final_eps | lr |
2
100%|| 50000/50000 [01:48<00:00, 459.70it/s]
| 1 | -883.0 | 1.265e+0 | 0.0376 | 0.01658 | 0.003183 |
100%|| 50000/50000 [01:48<00:00, 460.89it/s]
| 2 | -765.1 | 3.432e+0 | 0.03907 | 0.01284 | 0.00233 |
100%|| 50000/50000 [01:48<00:00, 459.75it/s]
| 3 | -1.036e+0 | 3.432e+0 | 0.1372 | 0.01463 | 0.0001 |
100%|| 50000/50000 [01:50<00:00, 454.25it/s]
| 4 | -1.163e+0 | 8.192e+0 | 0.1276 | 0.01956 | 0.0001 |
100%|| 50000/50000 [01:47<00:00, 464.04it/s]
| 5 | -919.4 | 3.146e+0 | 0.1056 | 0.01801 | 0.0001 |
100%|| 50000/50000 [01:49<00:00, 457.60it/s]
| 6 | -1.154e+0 | 3.707e+0 | 0.1979 | 0.01131 | 0.0001 |
100%|| 50000/50000 [01:48<00:00, 459.49it/s]
| 7 | -1.37e+03 | 8.73e+04 | 0.1744 | 0.01805 | 0.0001 |
=========================================================================
Iteration 0:
{'target': -882.9500152952969, 'params': {'bs': 12652.586464441809, 'eps': 0.03759859529433
Iteration 1:
{'target': -765.133347325027, 'params': {'bs': 34315.248936763615, 'eps': 0.039067682414731
Iteration 2:
{'target': -1036.4166837446392, 'params': {'bs': 34321.727982842254, 'eps': 0.1371758501866
Iteration 3:
{'target': -1163.2333531156182, 'params': {'bs': 81916.57958022068, 'eps': 0.12758175509872
Iteration 4:
{'target': -919.3666823580861, 'params': {'bs': 31464.259446718886, 'eps': 0.10556288295448
Iteration 5:
{'target': -1153.627796728164, 'params': {'bs': 37072.8998268652, 'eps': 0.1978587319007698
Iteration 6:
{'target': -1369.777799680829, 'params': {'bs': 87297.49885640763, 'eps': 0.174395020688153
Max {'target': -765.133347325027, 'params': {'bs': 34315.248936763615, 'eps': 0.03906768241
print('Max', optimizer.max)
Max {'target': -765.133347325027, 'params': {'bs': 34315.248936763615, 'eps': 0.03906768241
