import re
import json
import os
from multiprocessing.dummy import Pool
from threading import Lock
from itertools import repeat

import pandas
from fuzzywuzzy import fuzz
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
synset_lock = Lock()

def main():
    print("dawn")
    config = get_config()
    input_data = get_data(config)
    relevant_data = get_relevant_data(input_data, config)
    recommendation = recommend(relevant_data.get('users'), relevant_data.get('services'), config)
    write_data(recommendation, config)
    print("dusk")

def get_config():
    my_path = os.path.abspath(os.path.dirname(__file__))
    filename = my_path+"/config.json"
    with open(filename) as f:
        return json.load(f)

def get_data(config):
    return pandas.read_excel(config.get('file_name'),[config.get('user_sheet'), config.get('service_sheet')])

def get_relevant_data(data, config):
    output = {}
    user_df = data[config.get('user_sheet')]
    service_df = data[config.get('service_sheet')]
    output['users'] = user_df[config.get('user_columns')].assign(id = user_df[config.get('user_id_columns')].apply(lambda x: '_'.join(x.astype(str)), axis = 1)).to_dict(orient='records')
    output['services'] = service_df[config.get('service_columns')].assign(id = service_df[config.get('service_id_columns')].apply(lambda x: '_'.join(x), axis = 1)).to_dict(orient='records')
    return output

def recommend(users, services, config):
    pool = Pool(20)
    return pool.starmap(user_run, zip(users, repeat(services), repeat(config)))

def user_run(user, services, config):
    scores = recommendation_score(user, services, config)
    filtered_score = sort_and_filter(user, scores, config)
    print(filtered_score)
    return filtered_score


def sort_and_filter(user, score, config):
    output = {'user' : user.get('id')}
    count = 1
    for key, value in sorted(score.items(), key=lambda x: x[1], reverse=True):
        if count > config.get('max_recommendation'):
            break
        if value > config.get('threshold_score'):
            output['partner'+str(count)] = key
            output['partner_score'+str(count)] = round(value, 3)
            count = count + 1
    return output

def recommendation_score(user, services, config):
    user_service_score = {}
    matching_columns = config.get('matching_columns')
    user_additional_columns = config.get('user_additional_columns')
    matching_columns_weight = (1 / len(matching_columns))/(0.5*(1 / len(matching_columns))*len(set(matching_columns.values()).union(set(config.get('service_additional_columns'))))+1)
    user_additional_columns_weight = 0.5*(1 / len(matching_columns))/(0.5*(1 / len(matching_columns))*len(set(matching_columns.values()).union(set(config.get('service_additional_columns'))))+1)
    
    for service in services:
        score = config.get('zero_weight')
        for key in matching_columns:
            score = score + matching_columns_weight*match_text(config, user.get(key), service.get(matching_columns.get(key)))
        for u in user_additional_columns:
            for s in set(matching_columns.values()).union(set(config.get('service_additional_columns'))):
                score = score + user_additional_columns_weight*match_text(config, user.get(u), service.get(s))
        user_service_score[service.get('id')] = score
    return user_service_score

def match_text(config, datapoint1, datapoint2):
    if datapoint1 == None or datapoint2 == None or pandas.isna(datapoint1) or pandas.isna(datapoint2):
        return config.get('zero_weight')
    if datapoint1 == datapoint2:
        return config.get('one_weight')
    if get_text_matching_ratio(datapoint1, datapoint2) > config.get('text_matching_threshold'):
        return config.get('text_matching_weight')
    datapoint1 = list(filter(lambda w: w is not None and not w in stop_words, re.split(config.get('regex_delimiter'), datapoint1)))  
    datapoint2 = list(filter(lambda w: w is not None and not w in stop_words, re.split(config.get('regex_delimiter'), datapoint2))) 

    score=0
    dp_weight=1/(len(datapoint1))
    for d1 in datapoint1:
        for d2 in datapoint2:
            score = score + dp_weight*match_keywords(d1, d2, config)
            # return match_keywords(d1, d2, config)
    return score*config.get('delimited_weight')
    # return config.get('zero_weight')

def match_keywords(word1, word2, config):
    score = config.get('zero_weight')
    if word1.lower()==word2.lower():
        score = config.get('delimited_weight')
    elif get_text_matching_ratio(word1,word2) > config.get('text_matching_threshold'):
        score = config.get('delimited_text_matching_weight')
    else:
        d1_syn = get_synonyms(word1)
        d2_syn = get_synonyms(word2)
        if len(d1_syn.intersection(d2_syn)) > 0:
            score = config.get('synonym_weight')
    return score


def get_synonyms(word):
    synset_lock.acquire()
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
                synonyms.add(lm.name().lower())
    synset_lock.release()
    return synonyms

def get_text_matching_ratio(word1, word2):
    return fuzz.token_set_ratio(word1, word2)

def write_data(data, config):
    pandas.DataFrame(data).to_excel(config.get('output_file'))

if __name__ == '__main__':
    main()