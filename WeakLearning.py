{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww18620\viewh14680\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #This script uses the Snorkel package (https://www.snorkel.org/) to setup the weak learning process for developing an initial balanced training dataset \
\
\
#import the relevant packages\
\pard\pardeftab720\sl360\partightenfactor0

\f1\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
import pandas as pd\
import re\
from snorkel.labeling import labeling_function\
from snorkel.labeling import LFAnalysis\
from snorkel.labeling import PandasLFApplier\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
#read in the entire dataset of 14 million records\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 df=pd.read_csv("EntireData/entireData.csv")\
\
#fill any blanks in with string space\
df["body"].fillna(" ", inplace=True)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#setup the tags\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 df_train=df\
ABSTAIN=-1\
HAM=0\
SPAM=1\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#define the labeling functions\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 @labeling_function()\
def recruitCore(x):\
\'a0 \'a0search_list=['audition', 'salary', 'interview', 'earn money', 'high pay', 'scout']\
\'a0 \'a0return SPAM if re.compile('|'.join(search_list),re.IGNORECASE).search(x.body.lower()) else ABSTAIN\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 @labeling_function()\
def recruitBroad(x):\
\'a0 \'a0search_list=['salaries','recruiting', \'91staff\'92, \'91paid\'92, \'91employees\'92, \'91working\'92, \'91opportunity\'92, \'91earning\'92, \'91recruitment\'92, \'91recruiter\'92, \'91hiring\'92, \'91hire\'92]\
\'a0 \'a0return SPAM if re.compile('|'.join(search_list),re.IGNORECASE).search(x.body.lower()) else ABSTAIN\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 @labeling_function()\
def recruitEmbeddings(x):\
\'a0 \'a0search_list=['applicants', 'airfare', 'airfaretravel','renumeration', 'commission']\
\'a0 \'a0return SPAM if re.compile('|'.join(search_list),re.IGNORECASE).search(x.body.lower()) else ABSTAIN\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\
#apply the labeling functions to the data frame\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 lfs=[recruitCore, recruitBroad, recruitEmbeddings]\
applier=PandasLFApplier(lfs=lfs)\
L_train=applier.apply(df=df_train)\
print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#extract posts where label is likely recruitment\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 df1=df_train.iloc[L_train[:, 0] == SPAM]\
df2=df_train.iloc[L_train[:, 1] == SPAM]\
df3=df_train.iloc[L_train[:, 2] == SPAM]\
dfT=df1.append(df2)\
dfT=dfT.append(df3)\
dfR=dfT.drop_duplicates()\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#print size of the likely recruitment posts found\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("length of recruiting size:", len(dfR))\
dfR.to_csv('snorkelRec3RV1.csv')}