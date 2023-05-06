from peewee import Model
from connect_to_database import DB


class BaseModel(Model):
    """A base model that will use our Sqlite database."""

    class Meta:
        database = DB

