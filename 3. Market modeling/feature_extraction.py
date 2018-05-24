from utils import *
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval

# word embedings
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

class Features(object):

    def __init__(self, inputDict):
        self.inputs = inputDict
        
    def load_data(self):
        self.tweets = Features.load_tweets(self.tweets_path)
        self.prices = Features.load_prices(self.price_path, add_grid = True)
        
    def load_embeddings(self):
        # loads embeddings to dictionary
        self.embeddings = {}
        for item in self.embedding_path:
            path = self.embedding_path[item]
            if path[-4:] == '.bin':
                self.embeddings[item] = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            else:
                self.embeddings[item] = gensim.models.KeyedVectors.load_word2vec_format(path)
        
    @staticmethod
    def load_prices(path, add_grid = True):
        '''
        Loads prices from csv file.
        
        Returns dataframe with datetime index. Original prices from csv are placed on datetime grid
        with one minute frequency over oldest and newest price observations. This is done include After-Hours
        price changes - missing prices created by the grid are frontfilled by last valid observations.
        
        '''
        prices = pd.read_csv(path)
        prices['DateTime'] = prices['Date'] + ' ' + prices['Time']
        prices['DateTime'] = pd.to_datetime(prices['DateTime'])
        prices = prices.drop(['Date', 'Time', 'Volume'], axis=1)
        prices = prices.set_index('DateTime')
                     
        if add_grid:
            # Create grid
            grid_start = min(prices.index) - pd.DateOffset(days=5)
            grid_end = max(prices.index) + pd.DateOffset(days=5)
            grid = pd.date_range(start=grid_start, end=grid_end, freq='min')
            grid = pd.Series(grid).rename('DateTime')
            grid = pd.DataFrame(grid).set_index('DateTime')

            # Join grid with data
            prices = grid.join(prices)
            was_NaN = prices['Close'].isnull()
            prices = prices.fillna(method = 'ffill')
            prices['was_NaN'] = was_NaN
        return prices
    
    @staticmethod    
    def load_tweets(path):
        '''
        Loads preprocessed tweets from csv file.
        
        Returns multiindexed data frame with 'date', 'hour', '5min' ,'minute', 'id' index levels.
        Tweets with identical text occuring more than once per day are assumed to be spamm and are filtered.
        
        '''
        # Load data from csv and convert column lists of words
        tweets = pd.read_csv(path)
        tweets['lemmas'] = tweets['lemmas'].apply(literal_eval)
        tweets['tokens'] = tweets['tokens'].apply(literal_eval)

        # Create time variables
        tweets['date'] = tweets['created_at'].str[:10]
        tweets['hour'] = tweets['created_at'].str[11:13]
        tweets['minute'] = tweets['created_at'].str[14:16]
        tweets['5min'] = (tweets['minute'].astype(int)//5)*5
        
        # Spam filtering - Remove duplicate tweets in date
        tweets = tweets.drop_duplicates(['date', 'text'])
       
        # Drop redundant columns and index
        tweets = tweets.drop(['Unnamed: 0', 'created_at', 'text'], axis=1)
        tweets.set_index(['date', 'hour', '5min' ,'minute', 'id'], inplace = True)
        return tweets

    
    def create_corpuses(self):
        self.corpus = {}
        self.corpus_list = []
        
        for form in self.inputs['forms']:
            for agg in self.inputs['aggregates']:
                corpus_id = (form, agg)
                self.corpus_list.append(corpus_id)
                
                print ('Aggregating: '+ str(corpus_id))
                self.corpus[corpus_id] = aggregate_tweets(self.tweets, agg, form)
                
                
    def create_labels(self):
        self.label = {}
        self.label_list = []
        
        # Create list of label types
        self.label_type_list = []
        for direction in self.inputs['directions']:
            for window in self.inputs['windows']:
                label_type = (direction, window)
                self.label_type_list.append(label_type)        
        
        # Iterate over corpuses and label types
        for item in self.corpus_list:
            for label_type in self.label_type_list:
                label_id = item + label_type
                self.label_list.append(label_id)

                # Get direction of shift
                direction = label_type[0]
                window = label_type[1]                
                if direction == 'past':
                    window_dir = window
                elif direction == 'future':
                    window_dir = -1*window

                # Add label based on window to dataset
                self.label[label_id] = get_label(self.corpus[item], self.prices,  window_dir)
                    
    def create_BOW_datasets(self):
        self.BOW_dataset = {}
        self.BOW_dataset_list = []
        
        # Iterate over corpuses
        for item in self.corpus_list:
            for vec in self.inputs['BOW_vectorizers']:
                dataset_id = item + (vec,)
                self.BOW_dataset_list.append(dataset_id)
                
                # Vectorize text corpus
                text = self.corpus[item]['text']
                self.BOW_dataset[dataset_id] = BOW_vectorize(text, vec)

    def create_VW_datasets(self):
        self.VW_dataset = {}
        self.VW_dataset_list = []

        # Iterate over corpuses
        for item in self.corpus_list:
            for emb in self.inputs['embeddings']:
                for vec in self.inputs['WV_vectorizers']:
                    dataset_id = item + (emb, vec)
                    self.VW_dataset_list.append(dataset_id)

                    # Vectorize text corpus
                    text = self.corpus[item]['text']
                    embedding = self.embeddings[emb]
                    self.VW_dataset[dataset_id] = VW_vectorize(text, embedding, vec)                
                
            
    def create_BOW_links(self):
        self.BOW_link = {}
        self.BOW_link_list = []

        # Iterate over corpuses and label types
        for item in self.BOW_dataset_list:
            for label_type in self.label_type_list:
                link_id = item + label_type
                self.BOW_link_list.append(link_id)

                # Search for suitable label in self.label
                current_label_id = (item[0], item[1]) + label_type
                current_label = self.label[current_label_id]

                # Get array of indexes without NaN values
                index = current_label[current_label['Label'].notnull()].index
                self.BOW_link[link_id] = {'index': index, 'dataset_id': item, 'label_id': current_label_id}           

    def create_VW_links(self):
        self.VW_link = {}
        self.VW_link_list = []

        # Iterate over corpuses and label types
        for item in self.VW_dataset_list:
            for label_type in self.label_type_list:
                link_id = item + label_type
                self.VW_link_list.append(link_id)

                # Search for suitable label in self.label
                current_label_id = (item[0], item[1]) + label_type
                current_label = self.label[current_label_id]

                # Get array of indexes without NaN values
                index = current_label[current_label['Label'].notnull()].index
                self.VW_link[link_id] = {'index': index, 'dataset_id': item, 'label_id': current_label_id}                   
                
                
    def evaluate_BOW(self):
        self.BOW_predictions = {}
        self.BOW_results = {}
        
        # Iterate over dataset - label pairs
        for item in self.BOW_link_list:
            link = self.BOW_link[('lemmas', '5min', 'binary', 'future', 1)]
            
            # Extract dataset - label pair using links and shuffle 
            index = link['index']
            index = np.random.permutation(index)
            dataset = self.BOW_dataset[link['dataset_id']][index]
            label = self.label[link['label_id']].reindex(index)['Label']
            
            # Iterate over models
            for model in self.inputs['models']:
                
                # Calculate model predicitons
                prediction = get_model_prediction(dataset, label, model)
                prediction_id = item + (model,)
                self.BOW_predictions[prediction_id] = prediction
                
                # Calculate accuracy and kappa metrics
                kappa = cohen_kappa_score(label, prediction)
                accuracy = accuracy_score(label, prediction)
                
                result_id_kappa = item + (model, 'kappa')
                result_id_accuracy = item + (model, 'accuracy')
                
                self.BOW_results[result_id_kappa] = kappa
                self.BOW_results[result_id_accuracy] = accuracy
                
    def evaluate_VW(self):
        self.VW_predictions = {}
        self.VW_results = {}
        
        # Iterate over dataset - label pairs
        for item in self.VW_link_list:
            link = self.VW_link[('lemmas', '5min', 'binary', 'future', 1)]
            
            # Extract dataset - label pair using links and shuffle 
            index = link['index']
            index = np.random.permutation(index)
            dataset = self.VW_dataset[link['dataset_id']][index]
            label = self.label[link['label_id']].reindex(index)['Label']
            
            # Iterate over models
            for model in self.inputs['models']:
                
                # Calculate model predicitons
                prediction = get_model_prediction(dataset, label, model)
                prediction_id = item + (model,)
                self.VW_predictions[prediction_id] = prediction
                
                # Calculate accuracy and kappa metrics
                kappa = cohen_kappa_score(label, prediction)
                accuracy = accuracy_score(label, prediction)
                
                result_id_kappa = item + (model, 'kappa')
                result_id_accuracy = item + (model, 'accuracy')
                
                self.VW_results[result_id_kappa] = kappa
                self.VW_results[result_id_accuracy] = accuracy
                
                print('prediction_id')
                
    def save_VW_datasets(self):
        self.VW_dataset = {}
        self.VW_files = {}
        self.VW_dataset_list = []

        # Iterate over corpuses
        for item in self.corpus_list:
            for emb in self.inputs['embeddings']:
                path = self.embedding_path[emb]
                if path[-4:] == '.bin':
                    embedding = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
                else:
                    embedding = gensim.models.KeyedVectors.load_word2vec_format(path) 

                for vec in self.inputs['WV_vectorizers']:
                    dataset_id = item + (emb, vec)
                    self.VW_dataset_list.append(dataset_id)
                    print(dataset_id)
                    # Save vectorized text corpus as pickle
                    text = self.corpus[item]['text']
                    #embedding = self.embeddings[emb]
                    filename = str(item[0]) + '_' + str(item[1]) + '_' + str(emb) + '_' + str(vec) + '.pickle'            
                    self.VW_files[dataset_id] = filename
                    path = self.saving_location + '\\' + filename

                    with open(path, 'wb') as handle:
                        pickle.dump(VW_vectorize(text, embedding, vec), handle)
                
                
                
                
    def save_VW_datasets2(self):
        self.VW_dataset = {}
        self.VW_files = {}
        self.VW_dataset_list = []

        # Iterate over corpuses
        for item in self.corpus_list:
            for emb in self.inputs['embeddings']:
                for vec in self.inputs['WV_vectorizers']:
                    dataset_id = item + (emb, vec)
                    self.VW_dataset_list.append(dataset_id)
                    print(dataset_id)
                    # Save vectorized text corpus as pickle
                    text = self.corpus[item]['text']
                    embedding = self.embeddings[emb]
                    filename = str(item[0]) + '_' + str(item[1]) + '_' + str(emb) + '_' + str(vec) + '.pickle'            
                    self.VW_files[dataset_id] = filename
                    path = self.saving_location + '\\' + filename

                    with open(path, 'wb') as handle:
                        pickle.dump(VW_vectorize(text, embedding, vec), handle)
                        
    def load_embeddings(self):
        # loads embeddings to dictionary
        self.embeddings = {}
        for item in self.embedding_path:
            path = self.embedding_path[item]
            if path[-4:] == '.bin':
                self.embeddings[item] = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            else:
                self.embeddings[item] = gensim.models.KeyedVectors.load_word2vec_format(path) 

 
class Results(object):

    def __init__(self, path):
        self.path = path
        self.load_pickle()
        self.create_dataframe()
        
    def load_pickle(self):
        file = open(self.path,'rb')
        self.dict_results = pickle.load(file)
        file.close()
    
    def create_dataframe(self):
        self.dataframes = {}
        for i in self.dict_results:
            dataframe = Results.dict_to_dataframe(results[i])
            dataframe = dataframe.rename(columns = {'results':'run-' + str(i)})
            self.dataframes[i] = dataframe
            
        self.df = pd.concat([self.dataframes[i] for i in self.dataframes], axis = 1)    
            
    @staticmethod
    def dict_to_dataframe(input_dict):
        # Convert dictionary to dataframe
        dict_items = input_dict.items()
        df = pd.DataFrame(list(dict_items))

        # Add index
        index = pd.MultiIndex.from_tuples(df[0])
        df = df.drop(0, axis = 1)
        df = df.rename(columns = {1:'results'})
        df = df.set_index(index)
        return df