
# coding: utf-8

# In[1]:


conda install -c anaconda basemap


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import re
import datetime as dt
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error as mse
import sklearn.datasets
from sklearn.model_selection import train_test_split

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf
cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.corpora.dictionary import Dictionary
from nltk import FreqDist

from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity

import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap

import io
import requests


# # Loading our Data into Jupyter Notebooks

# In[3]:


okcupid = pd.read_csv("okcupid_profiles.csv")
okcupid.head()


# We check each variable to see which columns we want to focus on and analyze.

# In[4]:


okcupid.columns


# # Examining Location

# We want to study the distribution of okcupid users around the world to learn where this dating site is most popular, where most users reside and how we could use this information to adapt our data to best serve our analysis.
# 
# Our location variable in our okcupid data is formated such that the name of the city comes before a coma followed by either the state or country of that city. We make use of regex match to get rid of all the characters before the given state/country, that way our new variable $\bf{states}$ contains only the state\country, thus allowing us to examine the global user population.

# In[5]:


# We find the the best regex pattern to highlight the characters before 
# each state/country.
regx1 = r"^(.+?), california"

def show_regex_match(text, regex):
# Prints the string with the regex match highlighted.

    print(re.sub(f'({regex})', r'\033[1;30;43m\1\033[m', text))
    
show_regex_match("san francisco, california",regx1 ) # Test


# In[6]:


# We use string replace with regex to only keep the state/coutry
states = okcupid['location'].str.replace( r"^(.+?), ", "")
print('Number of states/countries in okcupid:',len(states.unique()))
states.unique()


# ## Visualization: Countplot

# Now we want to visualize the count of users per city in California to decide whether this is a more even distribution of geographic location in comparison to that of the world. 

# In[7]:


locations = states.to_frame()


# In[8]:


fig = plt.figure(figsize=(10,10))
fig.add_subplot(1,1,1)
ax = sns.countplot(y = "location", data = locations)
plt.xlim([0,40])
plt.title('Users per Location')


# ## Comparing Users in California to Users Anywhere Else

# Our count plot above demonstrates an extreme difference in the number of users between California and the rest of the world. According to our visualization there is a total of 59855 okcupid users in California and only 91 users anywhere else in the world.

# In[9]:


cali = locations[locations['location'] == 'california']
not_cali = locations[locations['location'] != 'california']
print('okcupid users worldwide:',len(okcupid['location']))
print('okcupid users in California:',len(cali))
print('okcupid users not in California',len(not_cali))


# The significantly small distribution of users worldwide compared to that in california allows us to make the decision to base the rest of our analysis and examination only with datasets from okcupid users that strictly reside in California. 

# In[10]:


# Once again we find the the best regex pattern to highlight all the locations with california in them.

regx1 = r"^(.+?), california"
def show_regex_match(text, regex):
    
#Prints the string with the regex match highlighted.
    
    print(re.sub(f'({regex})', r'\033[1;30;43m\1\033[m', text))

show_regex_match("san francisco, california",regx1 )


# We create a filter that will remove all of cases that are from users outside of California.

# In[11]:


filter = okcupid['location'].str.contains(r"^(.+?), california")
okcupid_ca = okcupid[filter]

print('lenght of new dataset:',len(okcupid_ca))
okcupid_ca.head()


# In[12]:


print('Number of Cities in CA:',len(okcupid_ca['location'].unique()))
okcupid_ca['location'].unique()


# ## Visualization: Countplot

# Now we want to visualize the count of users per city in California to decide whether this is a more even distribution of geographic location in comparison to that of the world. 

# In[13]:


fig = plt.figure(figsize=(10,20))
fig.add_subplot(1,1,1)
bx = sns.countplot(y = "location", data = okcupid_ca)
plt.xlim([0,1000])
plt.title('Users per CA City')


# Our count plot demonstrates a better distribution within the cities of california that will provide us with a better insight of the user demographics with respect to its region.

# ## Comparing OkCupid Users to Population Distribution Across California

# We wish to have a better visualization that will show the precise geographic location, distribution and magnitude of users per area within california. Hence, we download conda and import Basemap from mpl_toolkits.basemap which is a geographical visualization tool that takes in latitude and longitude coordinates to plot the population size within each region of a given area.
# 
# This visualization tool requires us to extract the latitude and longitude of each city inside of our okcupid_ca dataset and use this in our basemap projection to plot the population side within the given coordinates. To do so we will download a data frame "cities", that contain the names of each city in California with it's respctive coordinates.

# In[14]:


url="https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/data/california_cities.csv"
s=requests.get(url).content

cities = pd.read_csv(io.StringIO(s.decode('utf-8')))
cities.tail()


# Now that we have two seperate dataframes our goal is to merge these together by location and ensure that each city in our okcupid_ca dataframe is assigned their accurate coordinates. 
# 
# The variables that contain the location for each dataframe are $\bf{"city"}$for dataframe $\bf{"cities"}$ and $\bf{"location"}$ for the dataframe $\bf{"okcupid\_ca"}$, each formated as a string.
# 
# We notice that the strings in $\bf{"city"}$ for $\bf{"cities"}$ are not seperated by any spaces, thus to make each string of California cities exactly the same for each dataframe we utilize python's string methods to make each character lowercase, get rid of the spaces, commas, and the word california.
# 
# 

# In[15]:


okcupid_ca['location'] = (okcupid_ca['location']
 .str.lower()
 .str.strip()
 .str.replace('california', '')
 .str.replace(',', '')
 .str.replace(' ', '')
)


# We wish to make a new dataframe that contains the polished variable $\bf{"location"}$ in $\bf{"okcupid\_ca"}$ and a new column named $\bf{"users"}$ that contains the value counts of each city per location. We will name this new dataframe $\bf{"users\_okcupid"}$
# 
# As well, we will reset the index of $\bf{"users\_okcupid"}$ and rename the columns $\bf{"index"}$ by $\bf{"city"}$ and $\bf{"location"}$ by $\bf{"users"}$, so when we merge the data frames together we could call upon the column $\bf{"city"}$ on both frames for a smooth merge.

# In[16]:


users = okcupid_ca['location'].value_counts()
users_okcupid = users.reset_index().rename(index = str, columns = {"index":"city","location":"users"})
print('Number of CA cities in okcupid:',len(users_okcupid['city'].unique()))
users_okcupid['city'].unique()


# We repeat the same string method procedure to $\bf{"city"}$ in $\bf{cities}$.

# In[17]:


cities['city'] = (cities['city']
 .str.lower()
 .str.strip()
 .str.replace('california', '')
 .str.replace(',', '')
 .str.replace(' ', '')
)
print(len(cities['city'].unique()))
cities['city'].unique()


# *We* merge the dataframes together by mutual column $\bf{city}$

# In[18]:


okcupid_count = pd.merge(cities, users_okcupid , on="city")
okcupid_count.head()


# ## Visualization: Geographic Data Scatter Plot (Basemap)

# Now that we have a dataframe containing each CA city in okcupid with it's respective geographical coordinates we make use of Basemap toolkit to visualize the population distribution within California.
# 
# First, we assing variables for the values of latitude, longitude, total square area (Km${^2}$), and user counts in our new dataframe $\bf{okcupid\_count}$.

# In[19]:


lat = okcupid_count['latd'].values
lon = okcupid_count['longd'].values
total_users = okcupid_count['users'].values
area = okcupid_count['area_total_km2'].values


# Now, we plot our data into the basemap Geographical Scatter Plot.

# In[20]:


# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='l', 
            lat_0=37.5, lon_0=-119,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. Scatter city data, with color reflecting population and size reflecting area
m.scatter(lon, lat, latlon=True,
          c=(total_users), s=area,
          cmap='Reds', alpha=0.5)

# 3. Create colorbar and legend
plt.colorbar(label=r'User_Counts')
plt.clim(3, 7)

# 4. Make legend with dummy points
for a in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(a) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left');
plt.title('OkCupid Users in CA')


# From our visualization above we learn that okcupid users tend to reside mostly in rural cities accross the coast of California such as San Fransico, Los Angeles, and San Diego, meanwhile we see a lack of activity in suburban and western areas accross the state.

# # Missing Values

# At this point, we made the decision to remove NA values from our dataframe. Before deleting these rows, we compared the shape of the data before and after removing NA values. We decided that although we were only left with 7.4% of our data, we still had a big enough sample to proceed with.

# In[21]:


okcupid_ca.shape 


# In[22]:


okcupid_ca = okcupid_ca.dropna()
okcupid_ca.shape


# # Distribution of Gender

# As part of our preliminary analysis of the profiles, we examined the gender distribution between males and females on OkCupid. We found that about 56% of profiles were male, and 44% are female.

# In[23]:


okcupid_ca["sex"].value_counts()


# ## Visualization: Countplot

# In[24]:


fig = plt.figure(figsize = (10, 10))
sns.countplot(x = "sex", data = okcupid_ca)


# # Distribution of Age

# The second prelimiary analysis we looked at was the distribution of age. We can see that the majority of people on OkCupid are between the ages of 25 and 40, at an average of 33. The minimum age is 18 (you have to be 18 in order to register for an account), and the maximum is 69.

# In[25]:


okcupid_ca["age"].describe()


# In[26]:


fig = plt.figure(figsize = (20, 10))
sns.distplot(okcupid_ca["age"], bins = 40)


# # Distribution of Essay Response Polarity

# $\bf{Okcupid}$ gives its users the option to describe themselves and their lives through ten distinctive essays that capture their characteristics and live styles. Each essay corresponds to a question regarding different topics of interest that others can read to make judgements on that person based on them.
# 
# We wish to analyse the essays that best describe each user and appear to have the biggest impact in the decision making for others to decide whether they will be a good match or not. Through, the analysis of the polarity in each essay we will make our decision of what essays best contain the most positive descriptions of each user and use these in order to predict their best matches.
# 
# We will create a polarity data set from Vader lexicon text that contains the measurement of sentiment per character.

# ## Loading Polarity Data into Jupyter

# In[27]:


sent = pd.read_csv("vader_lexicon.txt", header = None, names = ["word", "polarity"], sep = "\t", index_col = [0], usecols = [0, 1])
sent.head()


# ## Isolating the Text in Each Essay

# From the 10 essays provided we decided to choose only 6 that appear to have the best descriptions and reliability of the characteristics and live styles of each person that best describes them.
# 
# We use string methods to polish each essay as to contain no other characters but those of words.

# In[28]:


okcupid_ca["My self"] = (okcupid_ca["essay0"]
                       .str.lower()
                       .str.replace("<br />\n", ""))
punct_re = r"[^\s\w]"
okcupid_ca['no_punc0'] = okcupid_ca["My self"].str.replace(punct_re, " ")


# In[29]:


okcupid_ca["My life"] = (okcupid_ca["essay1"]
                       .str.lower()
                       .str.replace("<br />\n", ""))
okcupid_ca['no_punc1'] = okcupid_ca["My life"].str.replace(punct_re, " ")


# In[30]:


okcupid_ca["Noticeable traits"] = (okcupid_ca["essay3"]
                       .str.lower()
                       .str.replace("<br />\n", ""))
okcupid_ca['no_punc3'] = okcupid_ca["Noticeable traits"].str.replace(punct_re, " ")


# In[31]:


okcupid_ca["Thoughts"] = (okcupid_ca["essay6"]
                       .str.lower()
                       .str.replace("<br />\n", ""))
okcupid_ca['no_punc6'] = okcupid_ca["Thoughts"].str.replace(punct_re, " ")


# In[32]:


okcupid_ca["Friday night"] = (okcupid_ca["essay7"]
                       .str.lower()
                       .str.replace("<br />\n", ""))
okcupid_ca['no_punc7'] = okcupid_ca["Friday night"].str.replace(punct_re, " ")


# In[33]:


okcupid_ca["Secrets"] = (okcupid_ca["essay8"]
                       .str.lower()
                       .str.replace("<br />\n", ""))
okcupid_ca['no_punc8'] = okcupid_ca["Secrets"].str.replace(punct_re, " ")


# ## Creating Tidy Formats

# We will convert each essay into dataframes labeled as $\bf{tidy\_formats}$ that split each word in each text and assign numerical values from zero to the lenght of the string correspoding to the position of each word.

# In[34]:


tidy_format0 = okcupid_ca["no_punc0"].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {"level_1":"num", 0:"word"})
tidy_format1 = okcupid_ca["no_punc1"].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {"level_1":"num", 0:"word"})
tidy_format3 = okcupid_ca["no_punc3"].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {"level_1":"num", 0:"word"})
tidy_format6 = okcupid_ca["no_punc6"].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {"level_1":"num", 0:"word"})
tidy_format7 = okcupid_ca["no_punc7"].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {"level_1":"num", 0:"word"})
tidy_format8 = okcupid_ca["no_punc8"].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {"level_1":"num", 0:"word"})


# ## Looking at the Polarity of Each Essay

# Tidy format allows a smooth transition to find the polarity of each essay and assing this into a new column.

# In[35]:


okcupid_ca['polarity0'] = (
    tidy_format0
    .merge(sent, how='left', left_on='word', right_index=True)
    .reset_index()
    .fillna(0)
    .groupby("index")
    .sum()["polarity"]
)
okcupid_ca[['My self', 'polarity0']].head()


# In[36]:


okcupid_ca['polarity1'] = (
    tidy_format1
    .merge(sent, how='left', left_on='word', right_index=True)
    .reset_index()
    .fillna(0)
    .groupby("index")
    .sum()["polarity"]
)
okcupid_ca[['My life', 'polarity1']].head()


# In[37]:


okcupid_ca['polarity3'] = (
    tidy_format3
    .merge(sent, how='left', left_on='word', right_index=True)
    .reset_index()
    .fillna(0)
    .groupby("index")
    .sum()["polarity"]
)
okcupid_ca[['Noticeable traits', 'polarity3']].head()


# In[38]:


okcupid_ca['polarity6'] = (
    tidy_format6
    .merge(sent, how='left', left_on='word', right_index=True)
    .reset_index()
    .fillna(0)
    .groupby("index")
    .sum()["polarity"]
)
okcupid_ca[['Thoughts', 'polarity6']].head()


# In[39]:


okcupid_ca['polarity7'] = (
    tidy_format7
    .merge(sent, how='left', left_on='word', right_index=True)
    .reset_index()
    .fillna(0)
    .groupby("index")
    .sum()["polarity"]
)
okcupid_ca[['Friday night', 'polarity7']].head()


# In[40]:


okcupid_ca['polarity8'] = (
    tidy_format8
    .merge(sent, how='left', left_on='word', right_index=True)
    .reset_index()
    .fillna(0)
    .groupby("index")
    .sum()["polarity"]
)
okcupid_ca[['Secrets', 'polarity8']].head()


# Here we can see the columns in our dataframe that show the polarity for each essay.

# In[41]:


okcupid_ca[["polarity0", "polarity1", "polarity3",  "polarity6", "polarity7", "polarity8"]].head()


# ## Fill NA Values

# We replace each NaN value with 0 to adapt our numberical values into a distribution plot

# In[42]:


okcupid_ca["polarity1"] = okcupid_ca["polarity1"].fillna(0)
okcupid_ca["polarity3"] = okcupid_ca["polarity3"].fillna(0)
okcupid_ca["polarity6"] = okcupid_ca["polarity6"].fillna(0)
okcupid_ca["polarity7"] = okcupid_ca["polarity7"].fillna(0)
okcupid_ca["polarity8"] = okcupid_ca["polarity8"].fillna(0)


# ## Visualization: Distribution Plot

# In[43]:


fig = plt.figure(figsize = (20, 10))
sns.distplot(okcupid_ca["polarity0"], label = 'My self', hist = False)
sns.distplot(okcupid_ca["polarity1"], label = 'My life', hist = False)
sns.distplot(okcupid_ca["polarity3"], label = 'Noticeable traits', hist = False)
sns.distplot(okcupid_ca["polarity6"], label = 'Thoughts', hist = False)
sns.distplot(okcupid_ca["polarity7"], label = 'Friday night', hist = False)
sns.distplot(okcupid_ca["polarity8"], label = 'Secrets', hist = False)
plt.xlim([-20,30])
plt.title("Distribution of Essay Polarity")
plt.xlabel("Polarity")


# In[44]:


polarity_description = okcupid_ca[["polarity0", "polarity1", "polarity3", "polarity6", "polarity7", "polarity8"]].describe()
polarity_description


# The distribution of polarity for each essay demonstrates that most texts tend to have a polarity near zero, however the texts in $\bf{"My self"}$ and $\bf{"My life"}$ appear to be the essays with the most positive distribution of polarity as shown in this plot.
# 
# This analysis to find the essays with the most positive polarity contributed to our desicion of basing our predictions on essay 0 ($\bf{"My self"}$) and essay 1 ($\bf{"My life"}$).

# # Looking at Most Frequently Used Words

# We will now look at the most frequently used words when people spoke about themselves and their lives. We created a new column, $\bf{info}$, which combined essay 0, $\bf{"My Self"}$, and essay 1, $\bf{"My Life"}$, together, stripping both from their html tags and any punctuation.

# In[45]:


okcupid_ca["info"] = okcupid_ca["essay0"].str.replace(r"<.*>", "") + okcupid_ca["essay1"].str.replace(r"<.*>", "")
okcupid_ca["info"].str.replace(r"\n", "").head()


# ## Preprocessing Essay Text

# In order to identify the key words in $\bf{info}$, we preprocessed the text by creating a new column called $\bf{key words}$ that contained all the text from $\bf{info}$, then converted all words to lower case, removed numbers and punctuation, tokenized, and removed stop words.

# In[46]:


stop_words = set(stopwords.words('english'))
punct = string.punctuation


# In[47]:


okcupid_ca['key_words'] = "" # Add a new column
okcupid_ca['key_words'] = np.nan

# Convert to lowercase and remove leading/trailing whitespace
okcupid_ca['key_words'] = (okcupid_ca['info']
 .str.lower()
 .str.strip()
)

# Remove Punctuation
punct_regex = r"[^\s\w]"
okcupid_ca['key_words'] = okcupid_ca['key_words'].str.replace(punct_regex, " ")

# Remove Numbers
num_regex = r"\d"
okcupid_ca['key_words'] = okcupid_ca['key_words'].str.replace(num_regex, " ")

# Extract the keywords (exclude stop words)
okcupid_ca['key_words'] = (okcupid_ca['key_words']
                       .apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])))

# Tokenize the esays using word_tokenize
okcupid_ca['key_words'] = (okcupid_ca['key_words'].apply(word_tokenize))


# ## Define a Dictionary of Keywords

# We now have located all the keywords in essays 0 and 1. We can use these keywords to create a dictionary and visualize the frequency distribution of the top used words. 

# In[48]:


# Processing Keywords
processed_keywords=[]
for keywords in okcupid_ca['key_words']:
    processed_keywords.append(keywords)
    

dictionary = Dictionary(processed_keywords)


# In[49]:


# Concatenate all keywords from all essays together
all_essay_words=[]
for word in processed_keywords:
    all_essay_words.extend(word) 


# ## Visualization: Bar Graph

# Now we will plot the most frequently used words in these two merged essays to figure out which words okcupid users tend to use most often to describe themselves.

# In[50]:


def plot_freq_words(words, terms = 10):

    fdist = FreqDist(words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.title('Most Frequently Used Words in OkCupid')
    plt.show()


# In[51]:


plot_freq_words(all_essay_words)


# From the bar graph, we can see the most frequently used words for all users are as expected for a dating site. The most used word is "love" and "like" followed by words that represent positive sentiments. This plot gives us a great insight into what most people are looking for in this dating site such as experiencing "new things" in "life" and meeting "good people".

# # Predicting OkCupid Matches

# Now that we have defined a dictionary of the words users use most frequently to describe themselves and their lives, we can now predict compatible OkCupid matches based on similarities in the language users chose to use in their essays. 

# In[52]:


corpus = [dictionary.doc2bow(keyword) for keyword in processed_keywords]
# Create the tf-idf model for the corpus 
tfidf = TfidfModel(corpus)

# Create the similarity data structure. 
# This is the most important part where we get the similarities between the essays.
sims = MatrixSimilarity(tfidf[corpus])


# Since we do not have any identification specific to an individual user, we will create a new column from the index labeled $\bf{user\_id}$ to assign each user an id. In this case, we decided to identify users by their row number.

# In[53]:


okcupid_ca = okcupid_ca.reset_index().rename(columns = {"index":"temp"})
okcupid_ca = okcupid_ca.reset_index().rename(columns = {"index":"user_id"})
okcupid_ca = okcupid_ca.drop("temp", axis = 1)
okcupid_ca.head()


# We have created a function that takes in any $\bf{user\_id}$, prints out the top words in their essay0 & essay1 along side their tf-idf score and the top 10 most similar matches for the given user. 

# In[54]:


def match_recommendation(user, dictionary, number_of_hits=10):
    top_words = 5
    # We will first start by  getting all the keywords related to the user's essay
    okcupid_match = okcupid_ca.loc[okcupid_ca.user_id==user] # get the user row
    keywords = okcupid_match['key_words']# Get all the keywords
    doc=[]
    for word in keywords:
        doc.extend(word)
    
    # Convert the doc into it's equivalent bag of words
    query_doc_bow = dictionary.doc2bow(doc)
    
    # convert the regular bag of words model to a tf-idf model where we have tuples
    # of the user ID and its tf-idf value for the essay
    query_doc_tfidf = tfidf[query_doc_bow]
    
    # get the array of similarity values between our user and every other user. 
    # To do this, we pass our list of tf-idf tuples to sims.
    similarity_array = sims[query_doc_tfidf] 
    # the length is the number of users we have. 

    similarity_series = pd.Series(similarity_array.tolist(), index=okcupid_ca.user_id.values) #Convert to a Series
    top_hits = similarity_series.sort_values(ascending=False)[1:number_of_hits+1] 
    #get the top matching results, i.e. most similar users; 
    # start from index 1 because every user is most similar to itself

    #print the words with the highest tf-idf values for the provided essay:
    sorted_tfidf_weights = sorted(tfidf[corpus[okcupid_match.index.values.tolist()[0]]], key=lambda w: w[1], reverse=True)
    print('Top %s words associated with this user by tf-idf are: ' % top_words)
    for term_id, weight in sorted_tfidf_weights[:top_words]:
        print(" '%s' (tf-idf score = %.3f)" %(dictionary.get(term_id), weight))
    print("\n")
    
    # Print the top matches
    print("Top %s most similar matches for user %s are:" %(number_of_hits, user))
    top_matches=[]
    for idx, (okcupid_match,score) in enumerate(zip(top_hits.index, top_hits)):
        print("%d %s (similarity score = %.3f)" %(idx+1, okcupid_match, score))
        top_matches.append(okcupid_match)
    return top_matches


# We tested our function out on user number $9$. We can see user $9$ has similarity score to user $1984$ of 15.6%.

# In[55]:


top_10_matches = match_recommendation(459, dictionary)


# # Distribution of Religion Associations

# Next we examined the distribution of religious associations. When setting up an OkCupid profile, users have the option of selecting the religious organization they associate with (or atheism or agnosticism) and specifying the degree of how religious they are. To start off, we wanted an overview of the various religious, stripping away to what degree of religious a person claims to be.

# In[56]:


okcupid_ca["religion"].head()


# ## Grouping Religions

# We defined a function, firstword(), that selects the first word from the string of words in the $\bf{religion}$ column. This allowed us to create a new column called $\bf{religions}$ that grouped all responses by religious organization.

# In[57]:


def firstword(text):
  '''
  Takes in a string of text, splits the string into a list of individual 
  words, and returns the first word.
  '''
  words = text.split()
  first = words[0]
  return first

firstword("agnosticism but not too serious about it") # Testing


# In[58]:


okcupid_ca["religions"] = okcupid_ca["religion"].apply(firstword)
okcupid_ca["religions"].head()


# ## Visualization: Countplot

# From the countplot below, we can see that the number one religion people identified with on their OkCupid profile was "other", followed closely by "agnosticism." This is not that surprising of a result, as all the OkCupid profiles we are looking at at this point are in California, and people tend to have increased moderate views on religion.

# In[59]:


fig = plt.figure(figsize = (20, 10))
sns.countplot(y = "religions", data = okcupid_ca)
plt.title('OkCupid Religions')


# # Distribution of Drug Usage

# Now we will look at people's responses when asked if they do drugs. About 78% of poeple say they never do drugs, 21% say they do drugs occassionally, and only 1.5% say they do drugs often. We noted that these responses may be biased, as people may not want to answer honestly to this sort of question on the internet.  

# In[60]:


okcupid_ca["drugs"].value_counts()


# # Examining the Relationship Between Drug Usage and Religion

# Next, we were curious to see if there was any relationship between drug usage and religion. 

# ## Quantifying Drug Usage Responses

# In order to examine linear relationship, we decided to quantify the three possible responses for drug usage. We defined a new column, $\bf{"druggie"}$, that includes to what level a person uses drugs. Someone who claimed to never use drugs recieved a 0, someone who claimed to sometimes use drugs received a 0.5, and someone who claimed to use drugs often recieved a 1.

# In[61]:


okcupid_ca["druggie"] = (okcupid_ca["drugs"]
                         .str.replace('sometimes', '.5')
                         .str.replace('often', '1')
                         .str.replace('never','0'))
okcupid_ca["druggie"].to_frame()
okcupid_ca["druggie"].value_counts()


# ## Defining Our Predictors and Response Variables

# In order to do prediction with religion, a categorical variable, we have to encode each value in the $\bf"religions"$ column as a binary vector indicating the non-numerical feature. We do this using dummy variables, which assigns a value of 0 or 1 to each column.

# In[62]:


# Religious organization is our predictor.
x = okcupid_ca["religions"]
x = pd.get_dummies(data = x, drop_first = True)
x.head()


# In[63]:


# Our response variable is how frequently users do drugs.
y = okcupid_ca["druggie"]
y.head()


# ## Linear Regression Model

# Now that we have defined our predictor and response, we can fit our linear regression model. First, we split our data into training data and testing data.  

# In[64]:


rel_tr, rel_te = train_test_split(x, test_size = .25)
print("Training Data Size: ", len(rel_tr))
print("Test Data Size: ", len(rel_te))


# Using the sklearn module in the LinearRegression package, we build a least squares model and fit the data.

# In[65]:


least_squares_model = sklearn.linear_model.LinearRegression()
least_squares_model.fit(x, y)


# Lastly, we calculate the root mean square error in our model. Mean square error is a measure of how close a fitted line is to our data points. The root mean square error is the distance, on average, of a data point from the fitted line, measured along a vertical line.

# In[66]:


rmse = np.sqrt((mse(y, least_squares_model.predict(x))))
print("Root Mean Square Error:",rmse )


# An RMSE of 0.22 is fairly high, indicating that religion is a weak predictor of drug usage.

# ## Grouping Degree of Religious Views

# Now, instead of looking at the specific religious organization users identify with, we look at the varying degrees of religious views. First, we looked at all the possible responses, and stripped away religious organization so all that was left was how religious a person claims to be.

# In[67]:


okcupid_ca["religion"].unique()


# We defined a dictionary to rate how religious a person is on a scale from -1 to 1. A person who claims to be very atheist recieves a score of -1, a person who claims to be agnostic recieves a score of 0, and a person who claims to be very religious (for any religious organization) recieves a score of 1. People who fell between the two extremes recieved positive or negative decimal values, depending on their response.

# In[68]:


d = {'agnosticism and very serious about it': 0,
     'catholicism but not too serious about it' : 0.25,
     'agnosticism and somewhat serious about it': 0, 
     'atheism': -0.75,
     'agnosticism and laughing about it' : 0, 
     'other' : 0,
     'christianity and somewhat serious about it' : 0.5,
     'atheism and laughing about it' : -.25,
     'catholicism and laughing about it' : 0.25,
     'judaism but not too serious about it' : 0.25,
     'other but not too serious about it' : 0,
     'hinduism but not too serious about it' : 0.25,
     'christianity but not too serious about it' : 0.25,
     'christianity and very serious about it' : 1,
     'other and somewhat serious about it' : 0,
     'buddhism but not too serious about it' : 0.25,
     'agnosticism but not too serious about it' : 0,
     'buddhism and laughing about it' : 0.25, 
     'other and laughing about it' : 0,
     'agnosticism' : 0, 
     'christianity' : 0.75,
     'catholicism and somewhat serious about it' : 0.5,
     'atheism but not too serious about it' : -.25,
     'other and very serious about it' : 0,
     'atheism and somewhat serious about it' : -.5, 
     'catholicism' : 0.75,
     'judaism and laughing about it' : 0.25,
     'atheism and very serious about it' : -1,
     'christianity and laughing about it' : 0.25, 
     'islam' : 0.75,
     'judaism and somewhat serious about it' : .5, 
     'buddhism' : 0.75,
     'hinduism and somewhat serious about it' : 0.5,
     'buddhism and somewhat serious about it' : 0.5,
     'islam and somewhat serious about it' : 0.5,
     'catholicism and very serious about it' : 1, 
     'judaism' : 0.75,
     'hinduism and very serious about it' : 1,
     'judaism and very serious about it' : 1,
     'buddhism and very serious about it': 1,
     'islam but not too serious about it' : 0.25,
     'islam and very serious about it' : 1,
     'hinduism and laughing about it' : 0.25, 
     'hinduism' : 0.75,
     'islam and laughing about it' : 0.25}


# We defined a new column called $\bf{"religious"}$ containing the transformed responses from $\bf{"religion"}$. From computing the mean of this column, we can see that on average, OkCupid users are slightly more religious than agnostic.

# In[69]:


okcupid_ca["religious"] = okcupid_ca["religion"].map(d)
okcupid_ca["religious"].mean()


# ## Visualization: Scatterplot

# In order to visualize our data, we convert columns $\bf{"religious"}$ and $\bf{"druggie"}$ into arrays.

# In[70]:


religion_array = okcupid_ca[["religious"]].values
religion_array


# In[71]:


drug_array = okcupid_ca["druggie"].astype(float).values
drug_array


# From our first attempt at creating a scatterplot, we noticed right away that there was too much overlapping to noticed any trends. We fix this by jittering our data.

# In[72]:


fig = plt.figure(figsize = (10, 10))
plt.scatter(religion_array, drug_array)


# In[73]:


okcupid_ca["druggie"] = okcupid_ca["druggie"].astype(float)
okcupid_ca["religious"] = okcupid_ca["religious"].astype(float)


# In[74]:


layout = dict(xaxis=dict(title="Do Religious Views Predict Drug Usage?"),yaxis=dict(title="How Often Do You Do Drugs?"))
jitter_y = okcupid_ca['druggie'] + 0.1 * np.random.rand(okcupid_ca['druggie'].size) -0.05
points = go.Scatter(x=okcupid_ca['religious'], y = jitter_y, 
                    mode="markers", 
                    marker=dict(opacity=0.5))
py.iplot(go.Figure(data=[points], layout=layout))


# We started by only jittering y, but were still not satisfied, so we created one last scatterplot, jittering both x and y.

# In[75]:


fig = plt.figure(figsize = (20, 10))
cm = plt.cm.get_cmap('winter')
jitter_y = okcupid_ca['druggie'] + 0.1 * np.random.rand(okcupid_ca['druggie'].size) -0.05
jitter_x = okcupid_ca['religious'] + 0.1 * np.random.rand(okcupid_ca['religious'].size) -0.05
points = plt.scatter(x = jitter_x, y = jitter_y, c = jitter_x, vmin = -1, vmax = 1, s = 1, cmap = cm)
plt.colorbar(points)
plt.xticks((-1, -.75, -.5, -.25, 0, .25, .5, .75, 1), ('Very Atheist', 'Atheist','Somewhat Atheist', 'Not Too Atheist',  'Agnostic', 'Not Too Religious', 
                   'Somewhat Religious', 'religious', 'Very Religious'))
plt.yticks((0, .5, 1), ('Never', 'Sometimes', 'Often'))
plt.xlabel("How Religious are You?")
plt.ylabel("How Often Do You Do Drugs?")
plt.yticks(rotation = 15)
plt.xticks(rotation = 15)
plt.title('Do Religious Views Predict Drug Usage?')
plt.show()


# From our scatterplot, we can see the majority of our users never user drugs, and of those users, the majority are agnostic. Users that sometimes use drugs are mainly agnostic as well, however there are more athetiest users that sometimes use drugs than religious users. Very few users said they use drugs often; we see these responses distributed similarly to the responses for never using drugs. It is important to note that these responses may be biased, in the case that users did not want to admit to using drugs on their online dating profile.
