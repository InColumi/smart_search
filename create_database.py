from models.context import Context
from models.ner import Ner
from models.cluster import Cluster
from models.test import Test
from models.centroid import Centroid

from connect_to_database import DB
from peewee import InternalError


def create_database():
    try:
        DB.connect()
        Context.create_table()
        Cluster.create_table()
        Ner.create_table()
        Centroid.create_table()
        Test.create_table()
    except InternalError as e:
        DB.rollback()
        print(e)
    finally:
        DB.close()

if __name__ == '__main__':
    print('Start create database')
    create_database()
    print('Database was created')
