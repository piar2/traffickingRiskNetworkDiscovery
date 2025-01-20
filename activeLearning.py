{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww19200\viewh33620\viewkind0
\deftab720
\pard\pardeftab720\sl360\partightenfactor0

\f0\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 #this file is the core active learning process that is run for each batch to retrain the predictive model and identify the next batch of posts #for labeling\
\
#import relevant packages\
import pandas as pd\
import os\
import tensorflow as tf\
import numpy as np\
from tensorflow import keras\
from tensorflow.keras.preprocessing.text import Tokenizer\
import tempfile\
import sklearn\
from sklearn.metrics import confusion_matrix\
from sklearn.model_selection import train_test_split\
#import tensorflow_addons as tfa\
#from tensorflow_addons.optimizers import AdamW\
from tensorflow.keras.preprocessing.sequence import pad_sequences\
import matplotlib\
import matplotlib.pyplot as plt\
import seaborn as sns\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #set the round for active learning to track batches and training\
ALRound=13\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#read in the entire dataset\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 entireData=pd.read_csv("EntireData/entireData.csv")\
\
#read in the labeled pool by the Oracle\
labeledData=pd.read_csv("ALBatches/labeledDataAfterB12.csv")\
\
#replace NANs\
dfDups=labeledData\
dfDups= dfDups.replace(np.nan, '', regex=True)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#store ids of the labeled data\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 list=dfDups.id\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#replace nan a cross entire data\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 entireData= entireData.replace(np.nan, '', regex=True)\
\
#remove labeled data from entire data to create the unlabeled pool data\
poolData=entireData[~entireData['id'].isin(list)]\
print("Entire data length:", len(entireData))\
print("Pool data length:", len(poolData))\
print("Labeled data length:", len(dfDups))\cb1 \
\cb3 poolData= poolData.replace(np.nan, '', regex=True)\
df=dfDups[['body', 'recLabel']].drop_duplicates()\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #Compute the positive label imbalance\
neg, pos = np.bincount(df['recLabel'])\
total = neg + pos\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #Compute the initial bias based on the label imbalance\
initial_bias = np.log([pos/neg])\
weight_for_0 = (1 / neg)*(total)/2.0\
weight_for_1 = (1 / pos)*(total)/2.0\cb1 \
\cb3 class_weight = \{0: weight_for_0, 1: weight_for_1\}\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #print('Weight for class 0: \{:.2f\}'.format(weight_for_0))\
#print('Weight for class 1: \{:.2f\}'.format(weight_for_1))\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#split out training, test and validation datasets \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 train_df, test_df = train_test_split(df, test_size=0.2)\
train_df, val_df = train_test_split(train_df, test_size=0.2)\
\
#select only the body and recruitment label for model training \
training_content=train_df['body']\
validation_content=val_df['body']\
training_labels=train_df['recLabel']\
validation_labels=val_df['recLabel']\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#set the vocabulary size, embeddings dimensions, length of posts, and truncation approach for keeping consistent lengths\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 vocab_size = 10000\
embedding_dim = 50\
max_length = 300\
trunc_type='post'\
pad_type='post'\
oov_tok = "<OOV>"\
num_epochs = 100\cb1 \
\cb3 tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)\
tokenizer.fit_on_texts(training_content)\
word_index = tokenizer.word_index\
\
# Pad the sequences so that they are all the same length\
training_sequences = tokenizer.texts_to_sequences(training_content)\
training_padded = pad_sequences(training_sequences,maxlen=max_length, truncating=trunc_type, padding=pad_type)\cb1 \
\cb3 validation_sequences = tokenizer.texts_to_sequences(validation_content)\
validation_padded = pad_sequences(validation_sequences,maxlen=max_length)\cb1 \
\cb3 training_labels_final = np.array(training_labels)\
validation_labels_final = np.array(validation_labels)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#setup the metrics to be computed for every epoch\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 METRICS = [\
\'a0 \'a0 \'a0 keras.metrics.TruePositives(name='tp'),\
\'a0 \'a0 \'a0 keras.metrics.FalsePositives(name='fp'),\
\'a0 \'a0 \'a0 keras.metrics.TrueNegatives(name='tn'),\
\'a0 \'a0 \'a0 keras.metrics.FalseNegatives(name='fn'),\
\'a0 \'a0 \'a0 keras.metrics.BinaryAccuracy(name='accuracy'),\
\'a0 \'a0 \'a0 keras.metrics.Precision(name='precision'),\
\'a0 \'a0 \'a0 keras.metrics.Recall(name='recall'),\
\'a0 \'a0 \'a0 keras.metrics.AUC(name='auc'),\
]\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#plot the key attributes for training and validation sets\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 matplotlib.use('Agg')\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 def plot_graphs(history, string, name):\
\'a0 plt.ioff()\
\'a0 fig=plt.figure()\
\'a0 plt.plot(history.history[string])\
\'a0 plt.plot(history.history['val_'+string])\
\'a0 plt.xlabel("Epochs")\
\'a0 plt.ylabel(string)\
\'a0 plt.legend([string, 'val_'+string])\
\'a0 fig.savefig(name)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
#Train core deep learning model with initialized bias and dropout\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!=")\
print("BIAS AND WEIGHTED MODEL W DROPOUT BEGINS HERE")\
print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!=")\
\
output_bias = tf.keras.initializers.Constant(initial_bias)\cb1 \
\cb3 lr_schedule=tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=1000, decay_rate=1, staircase=False)\cb1 \
\cb3 \
#define optimizer approach \
def get_optimizer():\
\'a0 \'a0return tf.keras.optimizers.Adam(lr_schedule)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 optimizer=get_optimizer()\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#define model layers\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 model = tf.keras.Sequential([\cb1 \
\cb3 \'a0 \'a0 tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),\
\'a0 \'a0 tf.keras.layers.GlobalAveragePooling1D(),\
\'a0 \'a0 keras.layers.Dropout(0.3),\
\'a0 \'a0 tf.keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)\
])\
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=METRICS)\
model.summary()\
#train model\
history = model.fit(training_padded, training_labels_final, epochs=num_epochs,validation_data=(validation_padded, validation_labels_final), \'a0class_weight=class_weight)\
\
#plot accuracy and loss for each round \
filename="coreImbalanceAccuracyRound" + str(ALRound) + ".png"\
plot_graphs(history, "accuracy", filename)\
filename="ALBatches/coreImbalanceLossRound" + str(ALRound) + ".png"\
plot_graphs(history, "loss", filename)\
\
#save trained model \
filename="baseImbalanceModel" + str(ALRound) + ".h5"\
model.save(filename)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=")\
print("Post uncertainty")\
print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!")\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 #predict on unlabeled pool from trained data\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 poolContent=poolData['body']\
poolSequences=tokenizer.texts_to_sequences(poolContent)\
poolPadded=pad_sequences(poolSequences, maxlen=max_length, truncating=trunc_type, padding=pad_type)\
predictions=model.predict(poolPadded)\
predDF=pd.DataFrame(\{'id': poolData.id, 'Prediction': predictions[:,0]\})\
#compute how many posts are \'93uncertain\'94 \
numUncertain=len(predDF[(predDF.Prediction>0.4) & (predDF.Prediction<0.6)])\
print("number in range 0.4-0.6:", numUncertain)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#save prediction distributions \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 filename="ALBatches/predictions" + str(ALRound) + ".csv"\
predDF.to_csv(filename)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=")\
print("Node uncertainty")\
print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!")\
#read in the data frame that has all locations associated with each post id (repeat id for every location)\
nodeSumID=pd.read_csv("TotalLocNodesByIDM.csv")\
nodeSumID=nodeSumID.fillna("")\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 entireContent=entireData['body']\
entireSequences=tokenizer.texts_to_sequences(entireContent)\
entirePadded=pad_sequences(entireSequences, maxlen=max_length, truncating=trunc_type, padding=pad_type)\
predictions=model.predict(entirePadded)\
predDFEntire=pd.DataFrame(\{'id': entireData.id, 'Prediction': predictions[:,0]\})\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #Filter to only posts we think are recruiting\
predDFSubset=predDFEntire[predDFEntire.Prediction>0.4]\
print("len entire subset >0,4", len(predDFSubset))\
\
#Filter the id-location df to only those ids with likely recruiting\
list=predDFSubset.id\
nodeSumSubset=nodeSumID[nodeSumID['id'].isin(list)]\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#merge the node attributes of each post with the original data frame of post predictions on the unlabeled pool \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 predDFSubset=pd.merge(nodeSumSubset, predDFSubset, on="id", how="outer")\
print("node and pred merge successful")\
print(predDFSubset.columns)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#tag whether a post is certain or uncertain \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 predDFSubset['certainFlag']=0\
predDFSubset['certainFlag'][predDFSubset['Prediction']>=0.8]=1\
predDFSubset['uncertainFlag']=0\
predDFSubset['uncertainFlag'][predDFSubset['Prediction']<0.8]=1\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #create node summaries by grouping\cb1 \
\cb3 nodeSum=predDFSubset.groupby(['city', 'region'], as_index=False).agg(\{'Prediction':'mean', 'certainFlag': 'sum', 'uncertainFlag':'sum'\})\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#compute node uncertainty score\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 nodeSum['scoreScaled']=nodeSum['uncertainFlag']*(nodeSum['uncertainFlag']/(nodeSum['certainFlag']+1))\
nodeSum=nodeSum.sort_values(by='scoreScaled', ascending=False)\
nodeSum=nodeSum.rename(columns=\{'Prediction':'avgNodePrediction', 'certainFlag': 'certainNodeTotal', 'uncertainFlag':'uncertainNodeTotal'\})\
filename="ALBatches/nodeSummary" + str(ALRound) + ".csv"\
nodeSum.to_csv(filename)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #now switch subset to focus on pooled data\
predDFSubset=predDF[predDF.Prediction>0.4]\
#add a few in the 0-0.3 range to check for new templates\
predDFUnlikely=predDF[predDF.Prediction<0.3]\
unlikelySample=predDFUnlikely.sample(n=200)\
predDFSubset=predDFSubset.append(unlikelySample)\
list=predDFSubset.id\
nodeSumSubset=nodeSumID[nodeSumID['id'].isin(list)]\
predDFSubset=pd.merge(nodeSumSubset, predDFSubset, on="id", how="outer")\
predDFSubset=pd.merge(predDFSubset, nodeSum, how='left', on=['city', 'region'])\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 IDScores=predDFSubset.groupby(['id'], as_index=False).agg(\{'Prediction':'mean', 'scoreScaled': 'mean'\})\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=")\
print("Edge uncertainty")\
print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!")\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 \outl0\strokewidth0 edgeSumID=pd.read_csv("TotalLocEdgesByIDM.csv")\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 edge\cf2 \cb3 SumSubset=\cf2 \cb3 edge\cf2 \cb3 SumID[\cf2 \cb3 edge\cf2 \cb3 SumID['id'].isin(list)]\
predDFSubset=pd.merge(edgeSumSubset, predDFSubset, on="id", how="outer")\
\
#create node summaries by grouping\cf2 \cb1 \
\cf2 \cb3 edge\cf2 \cb3 Sum=predDFSubset.groupby(['city_1\'92, 'region_1\'92, \'91city_2\'92, \'91region_2\'92], as_index=False).agg(\{'Prediction':'mean', 'certainFlag': 'sum', 'uncertainFlag':'sum\'92\})\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 #compute edge uncertainty score\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 edge \cf2 \cb3 Sum['\cf2 \cb1 edgeS\cf2 \cb3 coreScaled']= \cf2 \cb1 edge\cf2 \cb3 Sum['uncertainFlag']*(\cf2 \cb1 edge\cf2 \cb3 Sum['uncertainFlag']/(\cf2 \cb1 edge\cf2 \cb3 Sum['certainFlag']+1))\
\cf2 \cb1 edge\cf2 \cb3 Sum=\cf2 \cb1 edge\cf2 \cb3 Sum.sort_values(by='\cf2 \cb1 edge\cf2 \cb3 ScoreScaled', ascending=False)\
\cf2 \cb1 edge\cf2 \cb3 Sum=\cf2 \cb1 edge\cf2 \cb3 Sum.rename(columns=\{'Prediction':'avg\cf2 \cb1 Edge\cf2 \cb3 Prediction', 'certainFlag': 'certain\cf2 \cb1 Edge\cf2 \cb3 Total', 'uncertainFlag':'uncertain\cf2 \cb1 Edge\cf2 \cb3 Total'\})\
filename="ALBatches/\cf2 \cb1 edge\cf2 \cb3 Summary" + str(ALRound) + ".csv"\
edgeSum.to_csv(filename)\
\
predDFSubset=pd.merge(predDFSubset, edgeSum, how='left', on=['city_1\'92, 'region_1\'92, \'91city_2\'92, \'91region_2\'92])\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 IDScores=predDFSubset.groupby(['id'], as_index=False).agg(\{'Prediction':'mean', 'scoreScaled': 'mean\'92, \'91edgeScoreScale\'92:\'92mean\'92\})\cf2 \cb1 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=")\
print("Batch selection")\
print("=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!==!=!=!=!=!=!=!=!=!=!=!")\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 #compute the post uncertainty \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 IDScores['postUncertainty'] =(1-(.5-IDScores['Prediction']).abs())\
\
#check if any node scores are NA\
print("na in final ID nodescores:", IDScores.scoreScaled.isna().sum())\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#determine batch size \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 batchSize=5000\
print("batchSize:", batchSize)\
#Update p based on batch calc of new labeled data accuracy\
p=.2\
certainBatchSize=int(p*batchSize)\
print("certainBatchSize:" , certainBatchSize)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 #check a certain amount of the likely recruitment posts\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 certain=IDScores[IDScores.Prediction>0.7]\
print("count of certain >0.7:", len(certain))\
if len(certain)<certainBatchSize:\
certainBatchSize=len(certain)\cb1 \
\cb3 certainSample=certain.sample(n=certainBatchSize)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#check a certain amount of the \'93uncertain\'94 recruitment posts\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 uncertainBatchSize=int(max(batchSize-certainBatchSize , 0))\
print("uncertainBatchSize", uncertainBatchSize)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 uncertain=IDScores[(IDScores.Prediction>0.48) & (IDScores.Prediction<0.7)]\
print("count uncertain 0.48-0.7:", len(uncertain))\
uncertain=uncertain.sort_values(['postUncertainty', 'scoreScaled\'92, \'91edgeScoreScaled\'92], ascending=[False, False, False])\
if len(uncertain)<uncertainBatchSize:\
uncertainBatchSize=len(uncertain)\
uncertainSample=uncertain.head(uncertainBatchSize)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#check some of the unlikely recruitment posts to \'93explore\'94 and make sure we don\'92t miss templates \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 unlikely=IDScores[(IDScores.Prediction<0.48)&(IDScores.Prediction>0.4)]\
unlikelySample=unlikely.sample(n=200)\
print("count unlikely <0.48 and >0.4:", len(unlikely))\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#append the combination of likely, uncertain and unlikely posts to be labeled for the oracle \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 alBatch=certainSample.append(uncertainSample)\
alBatch=alBatch.append(unlikelySample)\cb1 \
\cb3 list=alBatch.id\
batchFromPool=poolData[poolData['id'].isin(list)]\
batchFromPool=pd.merge(batchFromPool, alBatch, on="id", how="left")\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#print out batch for labeling \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 filename="ALBatches/batch" + str(ALRound) + ".csv"\
batchFromPool.to_csv(filename)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("success!!")}