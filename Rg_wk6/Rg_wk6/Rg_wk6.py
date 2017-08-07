import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn
except ImportError:
    pass

sales = graphlab.SFrame('C:\\Machine_Learning\\Rg_wk6\\kc_house_data_small.gl\\')
(train_and_validation, test) = sales.random_split(.8, seed=1)
(train, validation) = train_and_validation.random_split(.8, seed=1)


#return 1+h0(xi)+h1(xi)+...., and output
def get_numpy_data(data_sframe,features,output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return (features_matrix,output_array)

#normalize features by 2-norm
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    feature_matrix = feature_matrix / norms
    return (feature_matrix, norms)

feature_list = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

#def getDist(a,b):
#    return np.sqrt(np.sum((a-b)**2))
#a = features_test[0]
#b = features_train[9]
#print getDist(a,b)
#for h in xrange(10):
#    dist = getDist(a,features_train[h])
#    print h,' = ',dist

def getDist(features_train,query):
    diff = features_train - query
    #print diff[-1].sum()
    #print np.sum(diff[15]**2)
    #print np.sum(diff**2,axis=1)[15]
    distance = np.sqrt(np.sum(diff**2,axis=1))
    return distance

distance = getDist(features_train,features_test[0])
print distance[100]

distance = getDist(features_train,features_test[2])
#get the index
i, = np.where( distance==np.min(distance))
print i
print np.min(distance)
print output_train[i]

def getDist_k(features_train,query,k):
    diff = features_train - query
    #print diff[-1].sum()
    #print np.sum(diff[15]**2)
    #print np.sum(diff**2,axis=1)[15]
    distance = np.sqrt(np.sum(diff**2,axis=1))
    distance = np.argsort(distance,axis=-1)
    return distance[0:k]

#query a single house
def getPrice(features_train,query,k):
    distance = getDist_k(features_train,query,k)
    pred = output_train[distance]
    #print pred
    #print np.average(pred)
    return np.average(pred)

print getPrice(features_train,features_test[2],4)

#query a set of house
def predict(features_train,query_set,k):
    result = []
    for q in query_set:
        pred = getPrice(features_train,q,k)
        result.append(pred)
    return result

pred_result = predict(features_train,features_test[0:9],10)

#Calculate rss
def get_rss(pred, output):
    rss = pred - output
    rss = rss * rss
    return rss.sum()

k_val = np.arange(1, 16, 1)
valid_rss = []
for i in k_val:
    pred_result = predict(features_train,features_valid,i)
    rss = get_rss(pred_result,output_valid)
    valid_rss.append(rss)

kvals = range(1, 16)
plt.plot(kvals, valid_rss,'bo-')
plt.show()

# k = 8 is the best one

pred_test = predict(features_train,features_test,8)
rss_test = get_rss(pred_test,output_test)
print rss_test


