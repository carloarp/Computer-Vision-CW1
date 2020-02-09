import numpy as np

import seaborn as sns
import scipy
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing


import numpy as np
import pandas as pd
import scipy.io as sc
import matplotlib as plt
import matplotlib.pyplot as plt

import sys
import time
import os
import psutil

from sklearn.neighbors import NearestNeighbors
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.vq import *

def print_image(face_image,title,mode):			# function for plotting an image
	if mode == 'no':
		return None
	face_image = np.reshape(face_image, (300,267))
	face_image = face_image.T
	plt.imshow(face_image, cmap='gist_gray')
	plt.title(title)
	plt.show()

def load_codebook(codebook_name, distortion_name, show):
	codebook_recall = np.load(codebook_name)
	distortion_recall = np.load(distortion_name)
	if show == 'yes':
		display_codebook(codebook_recall, distortion_recall)
	return codebook_recall, distortion_recall

def display_codebook(codebook, distortion):
	print("Codebook has shape",codebook.shape)
	print("Distortion =",distortion)
	return None

os.system('cls') 	
original_working_dir = os.getcwd()
dir = os.getcwd()						# gets the working directory of the file this python script is in
os.chdir (dir)							# changes the working directory to the file this python script is in

print("")								# also change plt.savefig to plt.show
np.random.seed(13)

### UNPACK MATLAB FILE
data = sc.loadmat('team_6.mat')			
random_selected_descriptors = data['random_selected_descriptors']
test_desc = data['descriptors_testing']
train_desc = data['descriptors_training'] 
train_idx = data['training_idx']
test_idx = data['testing_idx']

train_idx = train_idx.squeeze()
test_idx  = test_idx.squeeze()
#print("train_idx:\n", train_idx,"\n")

### ORGANIZE MATLAB DATA
ticks_train_images 			= train_desc[0]				# descriptor list for ticks train image
trilobite_train_images 		= train_desc[1]				# descriptor list for trilobite train image
umbrella_train_images 		= train_desc[2]
watch_train_images 			= train_desc[3]
waterlily_train_images 		= train_desc[4]
wheelchair_train_images 	= train_desc[5]
wildcat_train_images 		= train_desc[6]
windsorchair_train_images	= train_desc[7]
wrench_train_images 		= train_desc[8]
yinyang_train_images 		= train_desc[9]

ticks_train_idx				= train_idx[0].squeeze()	# index list for tick train image 
trilobite_train_idx			= train_idx[1].squeeze()	# index list for trilobite train image
umbrella_train_idx			= train_idx[2].squeeze()
watch_train_idx				= train_idx[3].squeeze()
waterlily_train_idx			= train_idx[4].squeeze()
wheelchair_train_idx		= train_idx[5].squeeze()
wildcat_train_idx			= train_idx[6].squeeze()
windsorchair_train_idx		= train_idx[7].squeeze()
wrench_train_idx 			= train_idx[8].squeeze()
yinyang_train_idx			= train_idx[9].squeeze()

ticks_test_images 			= test_desc[0]				# descriptor list for ticks test image
trilobite_test_images 		= test_desc[1]				# descriptor list for trilobite test image
umbrella_test_images 		= test_desc[2]
watch_test_images 			= test_desc[3]
waterlily_test_images 		= test_desc[4]
wheelchair_test_images 		= test_desc[5]
wildcat_test_images 		= test_desc[6]
windsorchair_test_images	= test_desc[7]
wrench_test_images 			= test_desc[8]
yinyang_test_images 		= test_desc[9]

ticks_test_idx				= test_idx[0].squeeze()		# index list for tick test image
trilobite_test_idx			= test_idx[1].squeeze()		# index list for trilobite test image
umbrella_test_idx			= test_idx[2].squeeze()
watch_test_idx				= test_idx[3].squeeze()
waterlily_test_idx			= test_idx[4].squeeze()
wheelchair_test_idx			= test_idx[5].squeeze()
wildcat_test_idx			= test_idx[6].squeeze()
windsorchair_test_idx		= test_idx[7].squeeze()
wrench_test_idx 			= test_idx[8].squeeze()
yinyang_test_idx 			= test_idx[9].squeeze()

number_of_images = 15

### LOAD HISTOGRAM DATA
print("LOADING HISTOGRAM DATA...")

accuracy_list 		= []
train_time_list 	= []
test_time_list 		= []



k_number = 120
codeword_list = [f'Codeword{i+1}' for i in range(0, k_number)]		# creates a list that contains codeword0...codeword'k'

output_dir_histogram_npy = str('histogram_npy_files_k='+str(k_number))
os.chdir(output_dir_histogram_npy)	

train_histogram = str('train_histogram_array_k='+str(k_number)+'.npy')
test_histogram = str('test_histogram_array_k='+str(k_number)+'.npy')

class_hist_list_train = np.load(train_histogram)
class_hist_list_test = np.load(test_histogram)

os.chdir(original_working_dir)

print("HISTOGRAM DATA LOADED: K =",k_number,"@",output_dir_histogram_npy,'\n')

### ORGANIZE HISTOGRAM DATA
ticks_train_histogram 			= class_hist_list_train[0]
trilobite_train_histogram 		= class_hist_list_train[1]
umbrella_train_histogram 		= class_hist_list_train[2]
watch_train_histogram 			= class_hist_list_train[3]
waterlily_train_histogram 		= class_hist_list_train[4]
wheelchair_train_histogram 		= class_hist_list_train[5]
wildcat_train_histogram 		= class_hist_list_train[6]
windsorchair_train_histogram 	= class_hist_list_train[7]
wrench_train_histogram 			= class_hist_list_train[8]
yinyang_train_histogram 		= class_hist_list_train[9]

ticks_test_histogram 			= class_hist_list_test[0]
trilobite_test_histogram 		= class_hist_list_test[1]
umbrella_test_histogram 		= class_hist_list_test[2]
watch_test_histogram 			= class_hist_list_test[3]
waterlily_test_histogram 		= class_hist_list_test[4]
wheelchair_test_histogram 		= class_hist_list_test[5]
wildcat_test_histogram 			= class_hist_list_test[6]
windsorchair_test_histogram 	= class_hist_list_test[7]
wrench_test_histogram 			= class_hist_list_test[8]
yinyang_test_histogram 			= class_hist_list_test[9]

combined_train_features = ticks_train_histogram.tolist() + trilobite_train_histogram.tolist() + umbrella_train_histogram.tolist() + watch_train_histogram.tolist() + waterlily_train_histogram.tolist() + wheelchair_train_histogram.tolist() + wildcat_train_histogram.tolist() + windsorchair_train_histogram.tolist() + wrench_train_histogram.tolist() + yinyang_train_histogram.tolist()

combined_test_features = ticks_test_histogram.tolist() + trilobite_test_histogram.tolist() + umbrella_test_histogram.tolist() + watch_test_histogram.tolist() + waterlily_test_histogram.tolist() + wheelchair_test_histogram.tolist() + wildcat_test_histogram.tolist() + windsorchair_test_histogram.tolist() + wrench_test_histogram.tolist() + yinyang_test_histogram.tolist()


### TRANSFORM TRAIN DATA INTO PANDAS DATAFRAME
print("COVERTING TRAIN DATA INTO PANDAS DATAFRAME...")

codeword_list 	= [f'Codeword{i}' for i in range(0, k_number)]

ticks_label 		= ['ticks']*number_of_images
trilobite_label 	= [f'trilobite']*number_of_images
umbrella_label 		= [f'umbrella']*number_of_images
watch_label 		= [f'watch']*number_of_images
waterlily_label 	= [f'waterlily']*number_of_images
wheelchair_label 	= [f'wheelchair']*number_of_images
wildcat_label 		= [f'wildcat']*number_of_images
windsorchair_label	= [f'windsorchair']*number_of_images
wrench_label 		= [f'wrench']*number_of_images
yinyang_label 		= [f'yinyang']*number_of_images

combined_train_label = ticks_label+trilobite_label+umbrella_label+watch_label+waterlily_label+wheelchair_label+wildcat_label+windsorchair_label+wrench_label+yinyang_label

df_column = codeword_list

train_features_array = combined_train_features

combined_train_df_column = codeword_list
combined_train_df_column.append("Label")
#combined_train_df = pd.DataFrame(columns=combined_train_df_column)
combined_train_df = pd.DataFrame(columns=["Codeword0","Codeword1","Codeword2","Label"])



ticks_train_df 			= pd.DataFrame(columns=df_column)
trilobite_train_df 		= pd.DataFrame(columns=df_column)
umbrella_train_df 		= pd.DataFrame(columns=df_column)
watch_train_df 			= pd.DataFrame(columns=df_column)
waterlily_train_df 		= pd.DataFrame(columns=df_column)
wheelchair_train_df 	= pd.DataFrame(columns=df_column)
wildcat_train_df 		= pd.DataFrame(columns=df_column)
windsorchair_train_df	= pd.DataFrame(columns=df_column)
wrench_train_df 		= pd.DataFrame(columns=df_column)
yinyang_train_df 		= pd.DataFrame(columns=df_column)

for K_train in range(0,k_number):
	ticks_train_df[codeword_list[K_train]] 			= ticks_train_histogram[:,K_train]
	trilobite_train_df[codeword_list[K_train]] 		= trilobite_train_histogram[:,K_train]
	umbrella_train_df[codeword_list[K_train]] 		= umbrella_train_histogram[:,K_train]
	watch_train_df[codeword_list[K_train]] 			= watch_train_histogram[:,K_train]
	waterlily_train_df[codeword_list[K_train]]	 	= waterlily_train_histogram[:,K_train]
	wheelchair_train_df[codeword_list[K_train]] 	= wheelchair_train_histogram[:,K_train]
	wildcat_train_df[codeword_list[K_train]] 		= wildcat_train_histogram[:,K_train]
	windsorchair_train_df[codeword_list[K_train]]	= windsorchair_train_histogram[:,K_train]
	wrench_train_df[codeword_list[K_train]] 		= wrench_train_histogram[:,K_train]
	yinyang_train_df[codeword_list[K_train]] 		= yinyang_train_histogram[:,K_train]
		
ticks_train_df["Label"] 		= ticks_label
trilobite_train_df["Label"] 	= trilobite_label
umbrella_train_df["Label"] 		= umbrella_label
watch_train_df["Label"] 		= watch_label
waterlily_train_df["Label"] 	= waterlily_label
wheelchair_train_df["Label"] 	= wheelchair_label
wildcat_train_df["Label"]		= wildcat_label
windsorchair_train_df["Label"]	= windsorchair_label
wrench_train_df["Label"] 		= wrench_label
yinyang_train_df["Label"] 		= yinyang_label
		
combined_train_df = combined_train_df.append(ticks_train_df)
combined_train_df = combined_train_df.append(trilobite_train_df)
combined_train_df = combined_train_df.append(umbrella_train_df)
combined_train_df = combined_train_df.append(watch_train_df)
combined_train_df = combined_train_df.append(waterlily_train_df)
combined_train_df = combined_train_df.append(wheelchair_train_df)
combined_train_df = combined_train_df.append(wildcat_train_df)
combined_train_df = combined_train_df.append(windsorchair_train_df)
combined_train_df = combined_train_df.append(wrench_train_df)
combined_train_df = combined_train_df.append(yinyang_train_df)

### TRANSFORM TEST DATA INTO PANDAS DATAFRAME
print("COVERTING TEST DATA INTO PANDAS DATAFRAME...")
codeword_list 	= [f'Codeword{i}' for i in range(0, k_number)]

#combined_test_label = ticks_label+trilobite_label+umbrella_label+watch_label+waterlily_label+wheelchair_label+wildcat_label+windsorchair_label+wrench_label+yinyang_label

df_column = codeword_list
combined_test_df_column = codeword_list
combined_test_df = pd.DataFrame(columns=combined_test_df_column)
ticks_test_df 			= pd.DataFrame(columns=df_column)
trilobite_test_df 		= pd.DataFrame(columns=df_column)
umbrella_test_df 		= pd.DataFrame(columns=df_column)
watch_test_df 			= pd.DataFrame(columns=df_column)
waterlily_test_df 		= pd.DataFrame(columns=df_column)
wheelchair_test_df	 	= pd.DataFrame(columns=df_column)
wildcat_test_df 		= pd.DataFrame(columns=df_column)
windsorchair_test_df	= pd.DataFrame(columns=df_column)
wrench_test_df 			= pd.DataFrame(columns=df_column)
yinyang_test_df 		= pd.DataFrame(columns=df_column)

for K_test in range(0,k_number):
	ticks_test_df[codeword_list[K_test]] 		= ticks_test_histogram[:,K_test]
	trilobite_test_df[codeword_list[K_test]] 	= trilobite_test_histogram[:,K_test]
	umbrella_test_df[codeword_list[K_test]] 	= umbrella_test_histogram[:,K_test]
	watch_test_df[codeword_list[K_test]] 		= watch_test_histogram[:,K_test]
	waterlily_test_df[codeword_list[K_test]] 	= waterlily_test_histogram[:,K_test]
	wheelchair_test_df[codeword_list[K_test]] 	= wheelchair_test_histogram[:,K_test]
	wildcat_test_df[codeword_list[K_test]] 		= wildcat_test_histogram[:,K_test]
	windsorchair_test_df[codeword_list[K_test]]	= windsorchair_test_histogram[:,K_test]
	wrench_test_df[codeword_list[K_test]] 		= wrench_test_histogram[:,K_test]
	yinyang_test_df[codeword_list[K_test]] 		= yinyang_test_histogram[:,K_test]
		
ticks_test_df["Label"] 			= ticks_label
trilobite_test_df["Label"] 		= trilobite_label
umbrella_test_df["Label"] 		= umbrella_label
watch_test_df["Label"] 			= watch_label
waterlily_test_df["Label"] 		= waterlily_label
wheelchair_test_df["Label"] 	= wheelchair_label
wildcat_test_df["Label"]		= wildcat_label
windsorchair_test_df["Label"]	= windsorchair_label
wrench_test_df["Label"] 		= wrench_label
yinyang_test_df["Label"] 		= yinyang_label
		
combined_test_df = combined_test_df.append(ticks_test_df, sort=True)
combined_test_df = combined_test_df.append(trilobite_test_df)
combined_test_df = combined_test_df.append(umbrella_test_df)
combined_test_df = combined_test_df.append(watch_test_df)
combined_test_df = combined_test_df.append(waterlily_test_df)
combined_test_df = combined_test_df.append(wheelchair_test_df)
combined_test_df = combined_test_df.append(wildcat_test_df)
combined_test_df = combined_test_df.append(windsorchair_test_df)
combined_test_df = combined_test_df.append(wrench_test_df)
combined_test_df = combined_test_df.append(yinyang_test_df)

#combined_test_df = combined_test_df.astype(int)

combined_df = pd.DataFrame(columns=combined_test_df_column)

combined_df = combined_df.append(combined_test_df, sort=True)
combined_df = combined_df.append(combined_train_df, sort=True)

train_df	= combined_train_df
test_df		= combined_test_df

print("")


### ===================================================================================================


import numpy as np
import pandas as pd
import random

from pprint import pprint

from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
from helper_functions import train_test_split, calculate_accuracy

random.seed(0)

def bootstrapping(train_df, n_bootstrap):	# creates a dataframe consisting of randomly picked training data
		
	bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
	df_bootstrapped = train_df.iloc[bootstrap_indices]
	
		
	return df_bootstrapped
		
def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
		
	forest = []
		
	for i in range(n_trees):
		df_bootstrapped = bootstrapping(train_df, n_bootstrap)
		tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
		forest.append(tree)
		
	return forest
		
def random_forest_predictions(test_df, forest):
	df_predictions = {}
	for i in range(len(forest)):
		column_name = "tree_{}".format(i)
		predictions = decision_tree_predictions(test_df, tree=forest[i])
		df_predictions[column_name] = predictions
		
	df_predictions = pd.DataFrame(df_predictions)
	random_forest_predictions = df_predictions.mode(axis=1)[0]
		
	return random_forest_predictions
		
		
df_bootstrapped = bootstrapping(train_df, n_bootstrap=5)

# control: k=120, trees=10, bootstrap=100, features=100, depth=10 

N_trees_list = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,500,600,800,900,1000,2000]
#N_trees_list = [10,20,30]

for item in N_trees_list:
	
	start_train_time = time.time()
	forest = random_forest_algorithm(train_df, n_trees=item, n_bootstrap=100, n_features=100, dt_max_depth=10)
	train_time = time.time() - start_train_time
		
	start_test_time = time.time()
	predictions = random_forest_predictions(test_df, forest)
	test_time = time.time() - start_test_time
		
	accuracy = calculate_accuracy(predictions,test_df.Label)
		
	accuracy_list.append(accuracy)
	train_time_list.append(train_time)
	test_time_list.append(test_time)
			
title_name = str('accuracy_k120_axis-aligned_varytrees2_bootstrap100_features100_depth10.png')

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(N_trees_list, train_time_list, '-r', label = 'train_time')
lns2 = ax.plot(N_trees_list, test_time_list, '-g', label = 'test_time')
ax2 = ax.twinx()
lns3 = ax2.plot(N_trees_list, accuracy_list, '-b', label = 'accuracy')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.set_xlabel("N_trees")
ax.set_ylabel("Time(s)")
ax2.set_ylabel("Accuracy(%)")

plt.savefig(title_name)
plt.show()
plt.close()

sys.exit()





















