from datetime import datetime
from models.context import Context
from models.ner import Ner
from models.cluster import Cluster
from nltk.corpus import stopwords
from connect_to_database import DB
from collections import defaultdict
from gensim.models import KeyedVectors
from filling_Meta_Ner_Context import insert_data, get_text_by_path
from sentence_transformers import SentenceTransformer, util
from config import settings

from models.test import Test
from models.centroid import Centroid
import statistics

import os
import re
import sys
import nltk
import spacy
import torch
import gensim.downloader as api
import peewee

from torch import Tensor
# from spacy.compat import cupy as cp
# import cupy as cp
# from thinc.api import set_gpu_allocator, require_gpu
# print(spacy.prefer_gpu(), require_gpu(0), require_gpu(1))

from peewee import fn, JOIN

def remove_articles(sentence):
    # Паттерн для определения артиклей
    pattern = r'\b(a|an|the)\b'

    # Удаление артиклей из предложения
    sentence_without_articles = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
    
    return sentence_without_articles

def clear_text(text: str, old, new):
    result = text
    while result.find(old) != -1:
        result = result.replace(old, new)
    return result.replace(old, new)

def review_to_words(text: str, stop_words):
    text = re.sub(r'[\n\t]', ' ', text) 
    text = remove_articles(text)
    letters_only = re.sub("[\.‚,+=:‘ˆ†;“”’–—\-_\"\\@!~#&$%*()+?|\{\}^`<>\/[\]]", " ", text)
    text = clear_text(letters_only, '  ', ' ')
    text = text.strip().lower()
    words = text.split(' ')
    words_without_stop_words = [w for w in words if w not in stop_words]
    return ' '.join(words_without_stop_words)
     



def get_clear_ner(ners, stop_words: set) -> list:
    return [(review_to_words(ner[0], stop_words), ner[1], ner[2]) for ner in ners]


def clust(context, model: SentenceTransformer, threshold):
    embeddings = model.encode(context,
                              batch_size=64,
                              show_progress_bar=True,
                              convert_to_tensor=True)
    clusters = util.community_detection(embeddings,
                                        min_community_size=1,
                                        threshold=threshold)
    return clusters, embeddings


def get_second_level_clusters(cluster_clusters: defaultdict, context_clusters: defaultdict, model: SentenceTransformer):
    second_level_clusters = []
    embed_list = []
    print('get_second_level_clusters. total_count: ', len(context_clusters.values()))
    count = 1
    for indexes, contexts in zip(cluster_clusters.values(), context_clusters.values()):
        if len(indexes) <= 3:
            continue
        try:
            item, embed = clust(contexts, model, 0.5)
            
        except Exception as e:
            print(str(e))
            continue
        if item != []:
            second_level_clusters.append([item, indexes])
            embed_list.append(embed)
        print(count)
        count += 1
    return second_level_clusters, embed_list


def calc_centroid(emb_group):
    # return sum(emb_group) / len(emb_group)
    
    sum_of_group = emb_group[0]
    for i in range(1, len(emb_group)):
        sum_of_group += emb_group[i]
    centroid = sum_of_group / len(emb_group)
    return centroid


def get_centroid_list(embed_list):
    return [calc_centroid(emb_group) for emb_group in embed_list]


def get_absolute_index(relative, emb_list):
    res_id = []
    res_emb = []
    for relative_item, emb_item in zip(relative, emb_list):
        list_id = []
        list_emb = []
        for index in relative_item[0]:
            list_id.append([relative_item[1][i] for i in index])
            list_emb.append([emb_item[i] for i in index])
        res_id.append(list_id)
        res_emb.append(list_emb)
    return res_id, res_emb


def get_important_words(cleared_ner, model2) -> list:
    important_words = []

    for sentence in cleared_ner:
        min_value = float("inf")
        index_min = -1

        for j, word in enumerate(sentence[0]):
            try:
                number = model2.get_vecattr(word, "count")
                if number < min_value:
                    min_value = number
                    index_min = j
            except Exception:
                important_words.append([word, sentence[1], sentence[2]])
                continue
        if index_min != -1:
            important_words.append([sentence[0][index_min], sentence[1], sentence[2]])
    return important_words


def get_new_clusters_by_important_words(important_words_for_new_ners: list) -> tuple:
    id_clusters = defaultdict(list)
    context_clusters = defaultdict(list)
    for important_word in important_words_for_new_ners:
        if important_word[0] != None and len(important_word[0]) > 2:
            id_clusters[important_word[0]].append(important_word[1])
            context_clusters[important_word[0]].append((important_word[2]))
    return id_clusters, context_clusters


def get_cos_dist(vector1, vector2):
    vector1 = vector1
    vector2 = vector2
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(vector1, vector2)


def get_min_cos_dist(context, centroid_list, model):
    embedding = model.encode(context, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
    count = 1

    for index, centroid in enumerate(centroid_list):
        vector1 = embedding
        vector2 = centroid

        cos_dist = get_cos_dist(vector1, vector2)

        if cos_dist < count:
            count = cos_dist
    return count


class Cluster_row:
    def __init__(self, word, vector, id_ner):
        self.word = word
        self.vector = vector
        self.id_ner = id_ner


def insert_new_cluster(cluster):
    str_centroid = [str(i) for i in cluster.vector.tolist()]
    str_centroid = ','.join(str_centroid)
    c = Cluster(word=cluster.word,
                centroid=str_centroid,
                size=1)
    id_cluster = Cluster.insert(word=c.word,
                                centroid=c.centroid,
                                size=c.size).execute()
    # print(id_cluster)
    Ner.update(cluster_id=id_cluster).where(Ner.id == cluster.id_ner).execute()

# def get_first_level_clusters(ners, model_for_clusters):
#      item, embed = clust(ners, model_for_clusters, 0.8)
#      print(item)



def insert_in_clusert(book_id: int, model_for_clusters: SentenceTransformer, model_for_imp_words, stop_words: set):
    print('Insert in table Cluster')
    table_ner = Ner.select(Ner.id, Ner.value, Ner.context_id)\
                .join(Context, JOIN.INNER)\
                .where(Ner.book_id == book_id)
    new_ner = [(ner.value, ner.id, ner.context_id.context) for ner in table_ner]
    
    # cleared_ner = get_clear_ner(new_ner, stop_words)
    
    # print('get_important_words')
    # important_words = get_important_words(cleared_ner, model_for_imp_words)
    # print('important_words', len(important_words))
    # print('get_new_clusters_by_important_words')
    # id_clusters, context_clusters = get_new_clusters_by_important_words(important_words)
    # print('id_clusters', len(id_clusters), 'context_clusters', len(context_clusters))
    # print('get_second_level_clusters')
    # print(context_clusters)
    # # return
    second_level_clusters, embed_list = get_second_level_clusters(id_clusters, context_clusters, model_for_clusters)
    print('second_level_clusters', len(second_level_clusters), len(embed_list))
    print('get_absolute_index')
    absolute_index, absolute_embed_list = get_absolute_index(second_level_clusters, embed_list)

    new_clusters = []

    id_cluster = Cluster.select(fn.MAX(Cluster.id)).scalar()
    if id_cluster is None:
        id_cluster = 0
    for word, clusters_list, emb_list_clusters in zip(id_clusters, absolute_index, absolute_embed_list):
        for cluster, emb_list_cluster, in zip(clusters_list, emb_list_clusters):
            for cluster_item, emb_item in zip(cluster, emb_list_cluster):
                new_clusters.append(Cluster_row(word, emb_item, cluster_item))
    print(len(new_clusters))
    data_for_udating = {}
    for cluster_item in new_clusters:
        table_cluster = Cluster.select().where(cluster_item.word == Cluster.word)
        if table_cluster.exists():
            max_cos_dist = 0
            max_cluster_id = 0
            centroid_of_max_cluster = 0
            max_row_size = 0
            max_vector = 0
            for row in table_cluster:
                vector = cluster_item.vector
                row_centroid = list(map(lambda i: float(i), row.centroid.split(',')))
                centroid_cluster = torch.Tensor(row_centroid)
                cos_dist = get_cos_dist(centroid_cluster, vector)
                if cos_dist >= 0.5 and max_cos_dist < cos_dist:
                    max_cluster_id = row.id
                    centroid_of_max_cluster = centroid_cluster
                    max_row_size = row.size
                    max_vector = vector
                    max_cos_dist = cos_dist

            if max_cos_dist:
                id_cluster = max_cluster_id
                new_size = max_row_size + 1
                new_centroid = (centroid_of_max_cluster * max_row_size + max_vector) / new_size
                str_centroid = list(map(lambda i: str(i), new_centroid.tolist()))
                str_centroid = ','.join(str_centroid)
                Cluster.update(centroid=str_centroid, size=new_size) \
                    .where(Cluster.id == id_cluster).execute()
                data_for_udating[cluster_item.id_ner] = id_cluster
            else:
                insert_new_cluster(cluster_item)

        else:
            insert_new_cluster(cluster_item)
    with DB.atomic():
        for key in data_for_udating:
            Ner.update(cluster_id=data_for_udating[key]).where(Ner.id == key).execute()


def main():
    try:
        try:
            DB.connect()
            print('Connection succes!')
        except:
            raise Exception('Connection error!')
        
        nlp = spacy.load("en_core_web_sm")

        try:
            stop_words = set(stopwords.words("english"))
        except Exception as e:
            print(str(e))
            nltk.download('stopwords')
            stop_words = set(stopwords.words("english"))
            
        try:
            model_for_clusters = SentenceTransformer('all-MiniLM-L6-v2')
            model_for_imp_words = KeyedVectors.load("glove-wiki-gigaword-200")
        except Exception as e:
            print(str(e))
            model_for_imp_words = api.load('glove-wiki-gigaword-200')
            model_for_imp_words.save("glove-wiki-gigaword-200")

        path = settings.path_to_texts
        files = os.listdir(os.path.join(path)) 
        files.sort(key=lambda x: int(x.split('.')[0]))
        count = 1

        isNext = False
        print('Start books')
        for file in files:
            try:
                print(f"Current book: {file}. Count: {count}")
                id_book = int(file.split('.')[0])
                if isNext:
                    if id_book == 82:
                        isNext = False
                    else:
                        count += 1
                        continue
                print(f'Book with id: {id_book} not exist in DataBase.')
                text = get_text_by_path(os.path.join(path, file))
                if not text:
                   print("Text is empty")
                   continue
                start_time = datetime.now()
                insert_data(text, nlp, id_book)
                print('Time inserting book in DataBase: {}'.format(datetime.now() - start_time))
                start_time = datetime.now()
                insert_in_clusert(id_book, model_for_clusters, model_for_imp_words, stop_words)
                print('Insert in cluster: {}'.format(datetime.now() - start_time))
                print('==================================================================')
                count += 1
                break
            except Exception as e:
                print(f"Error in {file}...")
                print(str(e))
                break
    except peewee.OperationalError as e:
        print('peewee.OperationalError')
        print(str(e))
    except Exception as e:
        print('Exception')
        print(str(e))
    except:
        print("Unexpected error:", sys.exc_info()[0])

def get_bunch_text(text, size):
    import math
    size_text = len(text)
    count_bunch = float(size_text / size)
    if count_bunch >= 1.0:
        return tuple(
            text[index * size: size + index * size] for index in range(math.ceil(count_bunch)))
    else:
        return [text]

def get_sentences(text, model_nlp):
    size_bench = 1000000
    bunchs_text = get_bunch_text(text, size_bench)
    sentences = []
    for bunch in bunchs_text:
        doc = model_nlp(bunch)
        tuple(sentences.append(sentence) for sentence in doc.sents if len(sentence) > 10)
    return sentences

def smart_cluster(sentences, model, id_book):
    clusters, embeddings = clust(sentences, model, 0.6)
    
    # for i in clusters:
    #     print(i)
    # print('-------------------------------------------------------------------------------------------')
    for cluster in clusters:
        embedding_cluster = [embeddings[i] for i in cluster]
        centroid = sum(embedding_cluster) / len(cluster)
        centroid_str = ','.join(list(map(lambda i: str(i), centroid.tolist())))
        centroid_db = Centroid(value=centroid_str)
        centroid_db.save()
        
        sentences_for_db = [sentences[i] for i in cluster]
        for sent in sentences_for_db:
            test = Test(context=sent, id_book=id_book, centroid_id=centroid_db.id)
            test.save()

def good_main():
    try:
        try:
            DB.connect()
            print('Connection succes!')
        except:
            raise Exception('Connection error!')
        
        nlp = spacy.load("en_core_web_sm")

        try:
            stop_words = set(stopwords.words("english"))
        except Exception as e:
            print(str(e))
            nltk.download('stopwords')
            stop_words = set(stopwords.words("english"))
            
        try:
            model_for_clusters = SentenceTransformer('all-MiniLM-L6-v2')
            model_for_imp_words = KeyedVectors.load("glove-wiki-gigaword-200")
        except Exception as e:
            print(str(e))
            model_for_imp_words = api.load('glove-wiki-gigaword-200')
            model_for_imp_words.save("glove-wiki-gigaword-200")

        path = settings.path_to_texts
        files = os.listdir(os.path.join(path)) 
        files.sort(key=lambda x: int(x.split('.')[0]))
        count = 1
        isNext = False
        print('Start books')
        for file in files:
            try:
                print(f"Current book: {file}. Count: {count}")
                id_book = int(file.split('.')[0])
                if isNext:
                    if id_book == 2944:
                        isNext = False
                    else:
                        count += 1
                        continue
                print(f'Book with id: {id_book} not exist in DataBase.')
                text = get_text_by_path(os.path.join(path, file))
                if not text:
                   print("Text is empty")
                   continue
                sentences = get_sentences(text, nlp)
                sentences = [review_to_words(str(s), stop_words) for s in sentences]
                # for i in sentences:
                #     print(i)
                print('=============================================')
                smart_cluster(sentences, model_for_clusters, id_book)
                count += 1
            except Exception as e:
                print(f"Error in {file}...")
                print(str(e))
                break
    except peewee.OperationalError as e:
        print('peewee.OperationalError')
        print(str(e))
    except Exception as e:
        print('Exception')
        print(str(e))
    except:
        print("Unexpected error:", sys.exc_info()[0])


if __name__ == "__main__":
    # main()
    good_main()
