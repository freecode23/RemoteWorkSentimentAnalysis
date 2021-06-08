from __future__ import division
import pandas as pd
import spacy
import string
import re, nltk
import nltk.corpus
from nltk.corpus import stopwords

from collections import Counter

def print_tweet(the_df, end_index):
    for i in range(0,end_index):
        print(i)
        print(the_df.iloc[i].tweet,'\n')
        
def print_empty_tweet(the_df):
    print(the_df[the_df.tweet == ""])

def delete_empty_tweet(the_df):
    indices_empty = []
    for i in range(len(the_df)):
        if(the_df.iloc[i].tweet == ""):
            indices_empty.append(i)
    
    the_df.drop(indices_empty, inplace = True)
    the_df.reset_index(drop = True, inplace =True)
    return the_df

def print_sentiment_tweet(the_df, sentiment = 'NEGATIVE'):
    for i in range(len(the_df)):
        if(sentiment == 'NEGATIVE'):
            if( the_df.iloc[i].VaderSentiment== 'NEGATIVE'):
                print(i)
                print(the_df.iloc[i].tweet)
        else:
            if( the_df.iloc[i].VaderSentiment== 'POSITIVE'):
                print(i)
                print(the_df.iloc[i].tweet)

# For VADER and BERT: Read learn more
# Remove all characters after the phrase read more, learn more, find our more, read our . Since these are usally just pointing out to resources
def remove_regex(the_regex, the_df):
    '''Remove the regex string from our tweets'''
    the_df = the_df.replace(to_replace=the_regex, value='', regex=True)
    return the_df

def remove_read_more(the_df):
    # define pattern:
    regex_read_our = '(?s)(?i)(read our)(?<=read our)(.*$)'
    regex_learn_more = '(?s)(?i)(learn more)(?<=learn more)(.*$)'
    regex_read_more = '(?s)(?i)(read more)(?<=read more)(.*$)'
    regex_find_out_more = '(?s)(?i)(find out more)(?<=find out more)(.*$)'
    
    the_df = remove_regex(regex_read_our,the_df)
    the_df = remove_regex(regex_find_out_more, the_df)
    the_df = remove_regex(regex_learn_more, the_df)
    the_df = remove_regex(regex_read_more, the_df)
    
    #drop nan
    the_df = delete_empty_tweet(the_df)
    the_df.dropna(inplace = True)
    the_df.reset_index(drop = True, inplace = True)
    # print_tweet(the_df, len(the_df))
    return the_df
                          
# 1. Remove End Hashtag
def remove_end_hashtag(the_df):  
    '''Remove hashtags at the end of every tweet(rows) in a dataframe'''
    regex_hashtag = '(#[A-Za-z0-9]+)'
    end_index = len(the_df)
    # create list to be appended to our df
    text_list = []
    # get text in each row
    for index_row in range(end_index):
        text = the_df.iloc[index_row].tweet
        
        # split the text
        text_split = text.split()
        last_index = -1     
        
        # get word in each list from the back
        for word in reversed(text_split):
            #if its a hashtag, remove it
            if(re.match(regex_hashtag,word) != None):
                text_split.pop(last_index)
            else:
                break
        # join back
        text = ' '.join(text_split)
        text_list.append(text)

    # update the_df
    the_df = the_df.assign(tweet=text_list)
    
    #drop nan
    delete_empty_tweet(the_df)
    the_df.reset_index(drop = True, inplace = True)
    
    return the_df


#2. Lemmatizing
def get_lemmatized_text(the_text):
    nlp = spacy.load('en_core_web_sm')
    the_text = nlp(the_text)
    
    the_text = ' '.join([word.lemma_ for word in the_text])
    return the_text

def lemmatized_df(the_df):
    #.apply wil return a series in this case
    the_df.tweet = the_df.tweet.apply(lambda x: get_lemmatized_text(x))

#3. Remove Special Char
def remove_char_from_text(the_text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,#!?\'\s]'
    return re.sub(pattern, ' ', the_text) 

def remove_char_vader(the_df):
    text_list = []
    for i in range(len(the_df)):                             
        # fix quote mark
        text = the_df.iloc[i].tweet.replace("’", "'")

        # fix hashtag position
        text = re.sub('# ', '#', text)

        # remove other characters
        text = remove_char_from_text(text)

        # remove extra whitespace
        text = re.sub(' +', ' ', text) 
    
        text_list.append(text)
    the_df = the_df.assign(tweet=text_list)
    #drop nan
    the_df.dropna(inplace = True)
    the_df.reset_index(drop = True, inplace = True)

    return the_df

# OPTIONAL
# 1. Split Hashtags
WORDS = nltk.corpus.brown.words()
COUNTS = Counter(WORDS)

def pdist(counter):
    "Make a probability distribution, given evidence from a Counter."
    N = sum(counter.values())
    return lambda x: counter[x]/N

P = pdist(COUNTS)

def Pwords(words):
    "Probability of words, assuming each word is independent of others."
    return product(P(w) for w in words)

def product(nums):
    "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
    result = 1
    for x in nums:
        result *= x
    return result

def splits(text, start=0, L=20):
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    return [(text[:i], text[i:]) 
            for i in range(start, min(len(text), L)+1)]

def segment(text):
    "Return a list of words that is the most probable segmentation of text."
    if not text: 
        return []
    else:
        candidates = ([first] + segment(rest) 
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=Pwords)


def get_capital_letter_index(the_word):
    "get the first index of the capital letter"
    for i in range(1, len(the_word)):
        if((the_word[i]).isupper()):
            return i
    return -1

def isAllCapital(the_word):
    "Check if a word is all capital letters"
    for i in range(1, len(the_word)):
        if(not(the_word[i]).isupper() and(the_word[i].isalpha())):
            return False
    return True
    

def get_split_word(the_word):
    'function that takes in 1 hashtag word, and convert to split words'
    final_word_list = []
    loose_char_list = []
    
    # CASE A: if all is capital return as is
    if(isAllCapital(the_word)):

        return the_word
    
    # CASE B: capital is in the middle, then split before the capital
    index_capital = get_capital_letter_index(the_word)
    if(index_capital != -1):
        string1 = the_word[0:index_capital]
        string2 = the_word[index_capital: len(the_word)]
        final_word_list = [string1, string2]
        return ' '.join(final_word_list)
    
    # CASE C: the word are not split by capital letter
    else:
        # 1. segment the word
        final_word_list = segment(the_word)  

        # 2. now we want to make sure word less than 3 chars is merged to previous word
        index = 1
        end_index = len(final_word_list)
        while(index < end_index):
            # 3. if length of current word is less than 2
            if( len(final_word_list[index]) <= 2):
                
                # 4. join current word to previous word
                final_word_list[index-1] = ''.join(final_word_list[(index-1):(index+1)])
                
                # 5. delete word at current
                final_word_list.pop(index)
                
                # 6. update end index after pop
                end_index = len(final_word_list)
            else:
                index += 1

    return ' '.join(final_word_list)   


def split_hashtag(the_df):
    'Split the hashtags in a given data frame'
    regex_hashtag = '#(\S*)'
    final_tweet_list = []
    indices_list = []
    # loop through each row and add modfied text to our final_tweet_list
    for i in range(len(the_df)):
        hashtag_word_dict = {}
        # 1. get list of merged words in a tweet: hashtag_word_list
        text = the_df.iloc[i].tweet
        hashtag_words_list = re.findall(regex_hashtag, text)      
        # 2. if there is merged words
        if(len(hashtag_words_list)!=0):
            print(i)  
            indices_list.append(i)
            print(hashtag_words_list)
            # for each word
            for i in range(len(hashtag_words_list)):
           
                # 3. get the word in each list, start from char 1 not 0. 0 is a symbol assign as key
                key_before_split = hashtag_words_list[i][0:len(hashtag_words_list[i])]
    
                # 4. split this word, assign as value
                value_after_split = get_split_word(key_before_split)
            
                # 5. create dict 
                hashtag_word_dict[key_before_split] = value_after_split
            
        # 6. remove hashtag symbol from the text
        text = text.replace('#', '')
        if(i == 1100):
            print(i)
            print(text)
        text_list = text.split(' ')
    
        # 7. loop through the text. if it finds the word in our key, replace with our value
        for i in range(len(text_list)):
            for key, value in hashtag_word_dict.items():
                if(text_list[i] == key):
                    text_list[i] = value
    
        text = ' '.join(text_list)
        # print_tweet(the_df, len(the_df))
        # remove extra whitespace
        text = re.sub(' +', ' ', text)
        final_tweet_list.append(text)

    # update df
    the_df.at[indices_list,'middle_hashtag'] = True
    the_df = the_df.assign(tweet=final_tweet_list)
    
    #drop nan
    the_df.dropna(inplace = True)
    the_df.reset_index(drop = True,inplace = True)

    return the_df

# 2. Remove Stopwords
def remove_stopwords(the_df):
    stop = stopwords.words('english')
    stop.remove('not')
    stop.extend(["click"])

    the_df.tweet = the_df.tweet.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #drop na
    the_df.dropna(inplace = True)
    the_df.reset_index(drop = True,inplace = True)

# 3. Final cleaning
def final_cleaning(the_df):
    text_list = []
    
    for i in range(len(the_df)):
        text = the_df.iloc[i].tweet
    
        # 1. remove repeat chars
        text = text.replace("...", ".")
        text = text.replace("..", ".")
        text = text.replace(",.", ".")
        text = text.replace(",,", ",")
    
        # 2. remove sapce after hashtag
        text = text.replace("# ", "#")
        text = text.replace('" ', '')
    
        # 3. remove quotes
        text = text.replace('"', '')
        text = text.replace("'", "")
    
        # 4. remove K or k
        text = re.sub(r'\b(k|K)\b', "", text)

        # 5. remove extra whitespace
        text = re.sub(' +', ' ', text) 
        text_list.append(text)
    
    # the_df['tweet'] = pd.DataFrame(text_list)
    the_df = the_df.assign(tweet=text_list)

    #drop na
    the_df.dropna(inplace = True)
    the_df.reset_index(drop = True)
    return the_df
                          

# 4. Remove Short tweets
def remove_short_tweets(the_df):
    '''Functions that remove tweets from df that are shorter than 4 words''' 
    indices_to_drop = []
    for i in range(len(the_df)):
        # 1. split text
        word_list = the_df.iloc[i].tweet.split()
    
        # 2. check len
        if(len(word_list) <= 4):
            indices_to_drop.append(i)
    
    the_df.drop(indices_to_drop, inplace = True)
    the_df.reset_index(drop = True,inplace = True)
    

    
# PIPELINE
def custom_pipeline(the_df,
                    del_split_hashtag = False, #optional
                    del_stopwords = True,  
                    del_short_tweets = False):

    # 1. split hashtag
    if(del_split_hashtag == True):
        print('split hashtag..')
        the_df = split_hashtag(the_df)  


    # 2. Remove Stopwords
    if(del_stopwords == True):
        print('Remove Stopwords..')
        remove_stopwords(the_df) 


    # 3. Final cleaning
    the_df = final_cleaning(the_df)

    
    # 4. Remove Short sentences
    if(del_short_tweets == True):
        print('Remove Short sentences..')
        remove_short_tweets(the_df)

    
    # 5. Remove duplicates
    the_df.drop_duplicates(subset=['tweet'], keep='first', inplace = True)
    the_df.reset_index(drop = True, inplace = True)

    return the_df
 
    

    

# 496
# Remote working can improve the #employeeexperience, but companies also benefit from adopting these arrangements by cutting costs, boosting #productivity and more. #FutureofWork #Business #RemoteWork #EX #WFH #WFA 

    

