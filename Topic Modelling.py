#!/usr/bin/env python
# coding: utf-8

# ## Extracting topics/themes from control descriptions

# In[55]:


# Data Structures
import numpy as np
import pandas as pd
import collections

# Utilities
import os
import warnings
warnings.filterwarnings('ignore')

# Visualizations 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS

# NLP pre-processing 
import nltk
import ssl
import gensim
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from gensim.parsing import preprocessing as pp
from gensim.corpora import Dictionary

stop_words = stopwords.words("english")
stop_words = set(stop_words)

# NLP models
from gensim.models import LdaModel
from gensim.models import CoherenceModel


# In[7]:


os.getcwd()


# ## Reading in the controls file

# In[10]:


controls = pd.read_csv('Controls.csv')


# In[11]:


controls.shape


# In[12]:


controls.columns.tolist()


# In[19]:


controls.isnull().sum()


# In[14]:


controls_sub = controls[['Control ID','Control Description']]


# In[15]:


controls_sub.shape


# In[18]:


controls_sub.isnull().sum()


# In[17]:


controls_sub.head()


# In[27]:


len(stop_words)


# ## Natural Language Processing 

# ### Step 1 : Preprocessing unstrucutred text

# In[28]:


pp_list = [
    lambda x: x.lower(),
    pp.strip_tags,
    pp.strip_multiple_whitespaces,
    pp.strip_punctuation,
    pp.strip_short
          ]

def tokenizer(line):
    """ Applies the following steps in sequence:
        Converts to lower case,
        Strips tags (HTML and others),
        Strips multiple whitespaces,
        Strips punctuation,
        Strips short words(min lenght = 3),
        --------------------------
        :param line: a document
        
        Returns a list of tokens"""
    
    tokens = pp.preprocess_string(line, filters=pp_list)
    return tokens


# In[42]:


get_ipython().run_cell_magic('time', '', "\ntrain_texts = []\n\nfor line in controls_sub[['Control Description']].fillna(' ').values:\n    train_texts.append(tokenizer(line[0]))#+' '+line[1]))")


# In[43]:


controls_sub['tokens'] = train_texts

controls_sub.head()


# ### Step 2 : Displaying most frequent terms

# In[44]:


unigram_counter = collections.Counter(x for xs in train_texts for x in set(xs))

for stop_word in stop_words:
    if stop_word in unigram_counter:
        unigram_counter.pop(stop_word)

unigram_counter.most_common(10)


# In[34]:


# For Mac
wc = WordCloud(background_color='white', random_state=42)

plt.figure(figsize=(10, 8))
plt.imshow(wc.fit_words(unigram_counter), interpolation='bilinear')
plt.axis("off")
#plt.savefig('../Data/Images/total_wordcloud.png')
plt.show()


# ### Step 3 : Extracting bi-grams and tri-grams

# In[35]:


get_ipython().run_cell_magic('time', '', 'bigram = gensim.models.Phrases(train_texts)\nbigram_phraser = gensim.models.phrases.Phraser(bigram)\ntokens_ = bigram_phraser[train_texts]\ntrigram = gensim.models.Phrases(tokens_)\ntrigram_phraser = gensim.models.phrases.Phraser(trigram)')


# ### Step 4 : Stemming and Lemmitizing 

# In[45]:


def process_texts(tokens):
    """Removes stop words, Stemming,
       Lemmatization assuming verb"""
    
    tokens = [token for token in tokens if token not in stop_words]
    tokens = bigram_phraser[tokens]
    tokens = trigram_phraser[tokens]
#     tokens = [stemmer.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return tokens


# In[46]:


get_ipython().run_cell_magic('time', '', '\nfinal_texts = []\n\nfor line in train_texts:\n    final_texts.append(process_texts(line))')


# In[47]:


controls_sub['final_tokens'] = final_texts
controls_sub.head()


# ### Step 5 : Creating a dictionary corpus

# In[49]:



np.random.seed(49)
# Create a dictionary(vocab) with all tokens
dictionary = Dictionary(final_texts)

# Filter tokens which appear less than 5 times,
# and those which appear more than 50% of the time.
dictionary.filter_extremes(no_below=2, no_above=0.5)

# Convert our documents to bag-of-words
corpus = [dictionary.doc2bow(text) for text in final_texts]


# In[50]:


dictionary_file_path = 'dictionary_controls_FULL.dict'
dictionary.save(dictionary_file_path)


# ## Topic Modelling

# In[52]:


def grid_lda(dictionary, corpus, texts, max_topics, min_topics=5, step=5,save=True,plot=True):
    np.random.seed(49)
    import time
    coherence_scores = []
    lda_list = []
    perplexity = []
    passes = 20
    iterations = 100
    eval_every = 50
    with open('log_LDA.txt', 'w') as f:
        for num_topics in range(min_topics, max_topics+1, step):
            print('#'*100)
            print('Training LDA with {} Topics'.format(num_topics))
            print()
            
            warnings.filterwarnings("ignore", category=DeprecationWarning) 
            start = time.time()
            lda =  LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)#,
                            #passes=passes,iterations=iterations,eval_every=eval_every)
            lda_list.append(lda)
            coherencemodel = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherencemodel.get_coherence()
            print('Coherence Score: ',coherence_score)
            coherence_scores.append(coherence_score)
            perplexity.append(lda.log_perplexity(corpus))
            print('Perplexity: ',perplexity[-1])
            print('Trained in {:0.3f}s'.format(time.time()-start))
            
            f.write('#'*100+' \n')
            f.write('Training LDA with {} Topics'.format(num_topics)+' \n') 
            f.write('Coherence Score: {}'.format(coherence_score)+' \n') 
            f.write('Perplexity: {}'.format(perplexity[-1])+' \n') 
            f.write('Trained in {:0.3f}s'.format(time.time()-start)+' \n')
            
            if save:
                lda.save('../Models/grid/{}_clusters_full_grid_active_score{:0.3f}.model'.format(num_topics,coherence_score))
                print('Model Saved under : ../Models/grid/{}_clusters_full_grid_active_score{:0.3f}.model'.format(num_topics,coherence_score))
                print()
        f.close() 
    if plot:   
        x = range(min_topics, max_topics+1, step)
        plt.plot(x, coherence_scores)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        #plt.legend(("coherence_values"), loc='best')
        plt.savefig('Coherence.png')
        plt.show()
        
        x = range(min_topics, max_topics+1, step)
        plt.plot(x, perplexity)
        plt.xlabel("Num Topics")
        plt.ylabel("Log Perplexity")
        plt.savefig('Perplexity.png')
        #plt.legend(("coherence_values"), loc='best')
        plt.show()
        
    return lda_list, coherence_scores, perplexity


# In[53]:


max_topics = 17
min_topics = 12

lda_list, coherence_values, perplexity = grid_lda(dictionary, corpus, final_texts, max_topics, min_topics, step=1,save=False,plot=True)


# In[57]:


np.argsort(coherence_values)[::-1][:15]+2


# In[59]:


get_ipython().run_cell_magic('time', '', "#list of number of clusters to try:\nclusters_n= [10,13,20,25,30]\n\n\n# Set random seed to reproduce results\nnp.random.seed(49)\n\nNUM_TOPICS = 15\n\n# Caution: the below parameters will take a long time to run\n# First run it with default parameters\npasses = 20\niterations = 400\neval_every = 50\n\n# Increase the number of passes to get better results. But it'll takes more time\nldamodel = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, \n                    passes=passes,\n                    iterations=iterations,\n                    eval_every=eval_every)")


# ## Visualizing the different topics and their terms

# In[60]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()

prepared_viz = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)


# In[61]:


import pyLDAvis

pyLDAvis.display(prepared_viz)


# In[62]:


from wordcloud import WordCloud

wc = WordCloud(background_color='white', max_font_size=100, width=600, height=400)

for t in range(ldamodel.num_topics):
    plt.figure(figsize=(6, 5))
    plt.imshow(wc.fit_words(dict(ldamodel.show_topic(t, 50))), interpolation='bilinear')
    plt.axis("off")
    plt.title("Topic #" + str(t+1))#topics[t])
    #plt.savefig('../Data/Full_Data/Images/total_wordcloud_topic{}_full_{}.png'.format(t+1,NUM_TOPICS))
    plt.show()

