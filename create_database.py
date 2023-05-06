from peewee import TextField, Model, PrimaryKeyField, \
                    CharField, IntegerField, \
                    ForeignKeyField, InternalError
from connect_to_database import DB
import json


class JSONField(TextField):
    """Class for json field"""

    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


class BaseModel(Model):
    """A base model that will use our Sqlite database."""

    class Meta:
        database = DB


class Meta_data(BaseModel):
    """Table with meta data"""
    id = PrimaryKeyField()
    gutenberg_id = CharField(max_length=10)
    author = CharField(null=True, max_length=1000)
    title = CharField(max_length=1000)
    json_data = JSONField()

    def __str__(self):
        return f"{self.id}, {self.gutenberg_id}, {self.author}, {self.title}, {self.json_data}"

    class Meta:
        table_name = 'Meta_data'


class Context(BaseModel):
    """Table with context info"""
    id = PrimaryKeyField()
    context = TextField()
    start = IntegerField()
    end = IntegerField()

    def __str__(self):
        return f"{self.id}, {self.context}, {self.start}, {self.end}"

    class Meta:
        table_name = 'Context'


class Cluster(BaseModel):
    id = PrimaryKeyField()
    word = CharField(index=True)
    centroid = CharField(max_length=8500)
    size = IntegerField()

    class Meta:
        table_name = 'Cluster'

    def __str__(self):
        return f"{self.id}, {self.centroid}, {self.size}"


class Ner(BaseModel):
    """Table with ner ifo"""
    id = PrimaryKeyField()
    context_id = ForeignKeyField(Context, related_name='context')
    book_id = ForeignKeyField(Meta_data, related_name='meta_data')
    cluster_id = ForeignKeyField(Cluster, related_name='cluster', null=True)
    value = CharField(max_length=500, index=True)
    start = IntegerField()
    end = IntegerField()
    type = CharField(max_length=15)

    def __str__(self):
        return f"{self.id}, {self.context_id}, {self.book_id}, {self.value}, {self.start}, {self.end}, {self.type}"

    class Meta:
        table_name = 'Ner'


def create_database():
    try:
        DB.connect()
        Meta_data.create_table()
        Context.create_table()
        Cluster.create_table()
        Ner.create_table()
    except InternalError as e:
        print(e)
    finally:
        DB.close()

if __name__ == '__main__':
    print('Start create database')
    create_database()
    print('Database was created')