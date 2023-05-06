from peewee import PostgresqlDatabase
from config import Settings

setings = Settings()

user = setings.user_name
password = setings.password
name_database = setings.name_database
address = setings.address
port = setings.port

DB = PostgresqlDatabase(name_database, user=user, password=password, host=address, port=port)
