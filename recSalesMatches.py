{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww19140\viewh33000\viewkind0
\deftab720
\pard\pardeftab720\sl360\partightenfactor0

\f0\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 #This file finds all of the matches from recruitment to sex-sales by looking for shared meta-data amongst the posts\
\
#import packages\
import pandas as pd\
import numpy as np\
import sys\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#read array input to run the code in parallel for computational efficiency \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 array=int(sys.argv[1])\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #read in labeled data pool\
filename="ALBatches/labeledDataAfterB13_Cat.csv"\
labeledData=pd.read_csv(filename, \'a0dtype=\{'feature.phone': str, 'feature.email':str, 'feature.username':str, 'feature.user_id': str, 'feature.website': str, 'feature.socialmedia':str, 'feature.location':str\})\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #filter labeled data for rec ads\
dfR=labeledData[labeledData.recLabel==1]\
recIDList=dfR.id\
\
#list of features to check for match\
featureList=["phone", "email", "username", "user_id", "socialmedia", "website", "location"]\
subFeatureList=["email", "username", "user_id", "socialmedia", "website"]\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #convert features from string to list, remove nulls\
for feature in featureList:\
	featureName="feature."+feature\
	try:\
		dfR[featureName].fillna("", inplace=True)\
	except:\
		print("isnull dfr replacement failure:", featureName)\
	pass\
	try:\
		dfR[featureName]=dfR[featureName].str.strip('[]').str.split(",")\
	except:\
		print("dfr split failure:", featureName)\
	pass\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #create location network empty data frame \
column_names=['id','city_1','region_1', 'lat_1', 'long_1', \'a0'city_2', 'region_2','lat_2','long_2', \'a0'freq', 'rel_type', 'category']\
locNetwork=pd.DataFrame(columns=column_names)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#create empty data frame to store id matches\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 idMatches=pd.DataFrame(columns=['id', 'idList'])\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 exceptions=0\
searchTermExceptions=0\
dfR['type']="Recruitment"\
allMatches=pd.DataFrame()\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 #pick range of files based on array, each file contains a sample of the overall data posts with 2713 files in total\
start=array*300-299\
end=start +299\
if end>2713:\
end=2713\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#go through each file\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 for f in range(start,end):\
	filename="EntireData/split/"+ str(f) + ".csv"\
	print(filename)\
#read in each data sample\
	df=pd.read_csv(filename, dtype=\{'feature.phone': str, 'feature.email':str, 'feature.username':str, 'feature.user_id': str, 'feature.website': str, 'feature.socialmedia':str, 'feature.location':str\})\
	df['type']=np.where(df.id.isin(recIDList), "RR", "RS")\
#fill in nans for data file\
	for feature in featureList:\
		featureName="feature."+feature\
		df[featureName].fillna("", inplace=True)\
		try:\
			df[featureName]=df[featureName].str.strip('[]').str.split(",")\
		except:\
		pass\
#go through each of the recruitment posts to find their matches in this datafile\
	for i in range(0, len(dfR)):\
#setup empty data frame for storing matches\
		matchDF=pd.DataFrame()\
#get first feature to check\
		featureName="feature."+featureList[0]\
#get meta data for first feature from recruitment posts and if it is empty, pass, otherwise search the sample for that same meta data\
		searchList=dfR.iloc[i][featureName]\
		id=dfR.iloc[i]['id']\
		category=dfR.iloc[i]['category']\
		searchListLen=len(searchList)\
		if (len(searchList)>0) and (searchList[0]!=''):\
			searchTerm=searchList[0]\
		else:\
			searchTerm="impossiblelist"\
		try:\
			temp=df[df[featureName].str.join('').str.contains(searchTerm)]\
			matchDF=matchDF.append(temp)\
		except:\
			searchTermExceptions=searchTermExceptions +1\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 #search for all of the items in the metadata list for the first feature\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 		if len(searchList)>0:\
			for j in range(1, searchListLen):\
			searchTerm=searchList[j]\
			try:\
				temp=df[df[featureName].str.join('').str.contains(searchTerm)]\
				matchDF=matchDF.append(temp)\
			except:\
				searchTermExceptions=searchTermExceptions +1\
\
#search for all the metadata associated with all of the features \
		for feature in subFeatureList:\
			featureName="feature."+feature\
			searchList=dfR.iloc[i][featureName]\
			for j in range(0, len(searchList)):\
				searchTerm=searchList[j]\
				if searchTerm != '':\
					try:\
						temp=df[df[featureName].str.join('').str.contains(searchTerm)]\
						matchDF=matchDF.append(temp)\
					except:\
						searchTermExceptions=searchTermExceptions +1\
\
#store all matches found with the same metadata as the recruitment post in the match data frame \
		matchDF=matchDF.loc[matchDF.astype(str).drop_duplicates().index]\
\
#if matches exist, then save them in IDMatches so we can check specific matches later, and then extract the location network\
	if len(matchDF)>0:\
		idList=matchDF.id\
		idListFormatted=pd.Series(idList).astype('str').values\
		idMatches=idMatches.append(\{'id':id, 'idList':idListFormatted\},ignore_index=True)\
		allMatches=allMatches.append(dfR.iloc[i], sort=False, ignore_index=True)\
		allMatches=allMatches.append(matchDF, sort=False, ignore_index=True)\
		try:\
			locStr=dfR.iloc[i]['locality']\
			locStr=locStr.replace("'", "\\"")\
			locList_1=eval(locStr)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
#for each location pair found from matching a recruitment post to sex sales post, increment a record for that location pair or add the new location pair if it does not exist \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 			for r in range(0, len(locList_1)):\
				location=locList_1[r]['label'].split(", ")\
				city_1=location[0]\
				region_1=location[1] if len(location)>1 else "NA"\
				lat_1=locList_1[r]['location']['lat']\
				long_1=locList_1[r]['location']['lon']\
				for m in range(0, len(matchDF)):\
					type=matchDF.iloc[m]['type']\
					locStr=matchDF.iloc[m]['locality']\
					locStr=locStr.replace("'", "\\"")\
					locList_2=eval(locStr)\
					for s in range(0, len(locList_2)):\
						location=locList_2[s]['label'].split(", ")\
						city_2=location[0]\
						region_2=location[1] if len(location)>1 else "NA"\
						lat_2=locList_2[s]['location']['lat']\
						long_2=locList_2[s]['location']['lon']\
						exist=len(locNetwork[(locNetwork['id']==id)& (locNetwork['city_1']==city_1) & 											(locNetwork['city_2']==city_2)])\
						if exist>0:\
							locNetwork['freq']=np.where((locNetwork.id==id) & 	(locNetwork.city_1==city_1)&(locNetwork.city_2==city_2)&(locNetwork.rel_type==type), locNetwork['freq']+1, locNetwork['freq'])\
						else:\
							new_row=\{'id': id, 'city_1':city_1, 'region_1':region_1, 'lat_1':lat_1, 'long_1':long_1, 'city_2':city_2, 'region_2':region_2, 'lat_2':lat_2, 'long_2':long_2, 'freq': 1, 'rel_type': type, 'category':category\}\
						locNetwork=locNetwork.append(new_row, ignore_index=True)\
#print("New record added")\
\
						except:\
						exceptions=exceptions+1\
\
\
\
#store the sample id matches for review\
idMatchesName="ALBatches/batch13IDMatches" + str(array) + ".csv"\
idMatches.to_csv(idMatchesName)\
allMatchesName="ALBatches/batch13AllMatches" + str(array)+".csv"\
allMatches.to_csv(allMatchesName)\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 \
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 print("total rec ads checked:", len(dfR))\
print("total num matched records:", len(allMatches))\
print("total exceptions:", exceptions)\
print("search term exceptions:", searchTermExceptions)\
\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb1 #store the location matches from the recruitment posts to the sex-sales posts\
\pard\pardeftab720\sl360\partightenfactor0
\cf2 \cb3 outputName="ALBatches/batch13LocMatches3Cat" + str(array) + ".csv"\
locNetwork.to_csv(outputName)}