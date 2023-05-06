from peewee import PostgresqlDatabase
from config import settings


DB = PostgresqlDatabase(database=settings.name_database, 
                        user=settings.user_name, 
                        password=settings.password, 
                        host=settings.address, 
                        port=settings.port)
