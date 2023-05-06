from models.context import Context
from models.ner import Ner
from models.cluster import Cluster
from connect_to_database import DB
from peewee import fn, chunked, JOIN

import math
import json
# from models.authors import Author
# from models.book_authors import Book_authors
# from models.titles import Titles
# def get_meta_data(id_book):
#     """Get meta data by id"""
#     fields = ('author', 'formaturi', 'language', 'rights', 'subject', 'title')
#     data = {}
#     for field in fields:
#         metadata = tuple(get_metadata(field, id_book))
#         data[field] = ['null'] if len(metadata) == 0 else metadata
#     return data

# def get_author_and_title(id_book: int): 

#     data = Author.select(Author.name.alias('author_name'), Titles.name.alias('title_name'))\
#                         .join(Book_authors, on=(Book_authors.ref_authors_id==Author.int_id), join_type=JOIN.INNER)\
#                         .join(Titles, on=(Titles.int_book_id==Book_authors.ref_book_id), join_type=JOIN.INNER)\
#                         .where(Book_authors.ref_book_id == id_book).dicts()
#     return data[:1]


def get_bench_text(text, size_bench):
    size = len(text)
    count_bench = float(size / size_bench)
    if count_bench >= 1.0:
        return tuple(
            text[index * size_bench: size_bench + index * size_bench] for index in range(math.ceil(count_bench)))
    else:
        return [text]


def create_ners(ners, id_ner, entity, max_id_context, max_id_meta_data, delta):
    if entity == []:
        return id_ner

    bad_type = ('CARDINAL', 'ORDINAL', 'TIME', 'QUANTITY', 'DATE', 'MONEY', 'LANGUAGE', 'PRODUCT')

    for el in entity:
        if el.label_ not in bad_type:
            ner = {'id': id_ner,
                   'context_id': max_id_context,
                   'book_id': max_id_meta_data,
                   'cluster_id': None,
                   'value': str(el),
                   'start': el.start_char + delta,
                   'end': el.end_char + delta,
                   'type': el.label_}
            id_ner += 1
            ners.append(ner)
    return id_ner


def insert_data(text: str, nlp, id_book: int):
    print('In insert_data')
    size_bench = 100000
    bench_text = get_bench_text(text, size_bench=size_bench)
    print("Count bench: ", len(bench_text), 'Size bench:', size_bench, " Size text: ", len(text))

    # max_id_meta_data = insert_to_meta_data(id_gutenberg)
    max_id_meta_data = id_book
    for index_bench, item_text in enumerate(bench_text):
        print(index_bench + 1)
        delta = index_bench * size_bench
        doc = nlp(item_text)

        current_id_context_table = Context.select(fn.MAX(Context.id)).scalar()
        current_id_context_table = 1 if current_id_context_table is None else current_id_context_table + 1

        current_id_ner_table = Ner.select(fn.MAX(Ner.id)).scalar()
        current_id_ner_table = 1 if current_id_ner_table is None else current_id_ner_table + 1

        context_for_inserting = []
        ners_for_inserting = []
        doc_sents = tuple(sentence for sentence in doc.sents)

        if index_bench == 0:
            context = {'id': str(current_id_context_table),
                       'context': ' '.join((str(i) for i in (doc_sents[0], doc_sents[1]))),
                       'start': str(doc_sents[0].start_char + delta),
                       'end': str(doc_sents[1].end_char + delta)}
            context_for_inserting.append(context)
            current_id_ner_table = create_ners(ners_for_inserting,
                                               current_id_ner_table,
                                               doc_sents[0].ents,
                                               current_id_context_table,
                                               max_id_meta_data,
                                               delta)
            current_id_context_table += 1

        for i in range(1, len(doc_sents) - 1):
            context = {'id': str(current_id_context_table),
                       'context': ' '.join((str(i) for i in (doc_sents[i - 1], doc_sents[i], doc_sents[i + 1]))),
                       'start': str(doc_sents[i - 1].start_char + delta),
                       'end': str(doc_sents[i + 1].end_char + delta)}
            context_for_inserting.append(context)
            current_id_ner_table = create_ners(ners_for_inserting,
                                               current_id_ner_table,
                                               doc_sents[i].ents,
                                               current_id_context_table,
                                               max_id_meta_data,
                                               delta)
            current_id_context_table += 1

        if index_bench == len(bench_text) - 1:
            context = {'id': str(current_id_context_table),
                       'context': ' '.join(
                           (str(i) for i in (doc_sents[len(doc_sents) - 2], doc_sents[len(doc_sents) - 1]))),
                       'start': str(doc_sents[len(doc_sents) - 2].start_char + delta),
                       'end': str(doc_sents[len(doc_sents) - 1].end_char + delta)}
            context_for_inserting.append(context)
            current_id_ner_table = create_ners(ners_for_inserting,
                                               current_id_ner_table,
                                               doc_sents[len(doc_sents) - 1].ents,
                                               current_id_context_table,
                                               max_id_meta_data,
                                               delta)
            current_id_context_table += 1

        with DB.atomic():
            for item in context_for_inserting:
                Context.insert(item).execute()

            for batch in chunked(ners_for_inserting, 999):
                Ner.insert_many(batch).execute()
            # for batch in chunked(context_for_inserting, 999):
            #     Context.insert_many(batch).execute()


def get_text_by_path(path: str):
    """Return text by path"""
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()
