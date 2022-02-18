import pandas as pd
import matplotlib.pyplot as plt

db_orcal = pd.read_csv('db_oracle_11g.csv')
container = pd.read_csv('dcos_container.csv')
docker = pd.read_csv('dcos_docker.csv')
mw_redis = pd.read_csv('mw_redis.csv')
os = pd.read_csv('os_linux.csv')

cmdb_group = db_orcal.groupby("cmdb_id")
db = cmdb_group.get_group('db_003')
db["time"] = pd.to_datetime(db.timestamp, unit='ms', origin='1970-01-01 08:00:00')
db = db.sort_values(by='timestamp')
metrics = db.groupby('name')

db_values = db[['name', 'timestamp', 'value']]

from tsfresh import extract_features
ext_features = extract_features(db_values, column_id='name', column_sort='timestamp')
#ext_features_t = ext_features.T
ext_features = ext_features.dropna(axis=1, how='any')
ext_features = ext_features.to_numpy()

from sklearn.decomposition import PCA
pca = PCA(n_components=30)
ld_features = pca.fit_transform(ext_features)
ld_features = ld_features.T
print(ld_features)

from castle.algorithms import PC
#from castle.algorithms import DirectLiNGAM

# structure learning
#pc = PC()
#pc.learn(ld_features)
#adj_matrix = pc.causal_matrix

#g = DirectLiNGAM()
#g.learn(metrics)
#adj_matrix = g.causal_matrix

#print(adj_matrix)
