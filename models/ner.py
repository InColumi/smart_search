from peewee import TextField, PrimaryKeyField, IntegerField, ForeignKeyField
from models.base import BaseModel
from models.context import Context
from models.cluster import Cluster


class Ner(BaseModel):
    """Table with ner ifo"""
    id = PrimaryKeyField()
    context_id = ForeignKeyField(Context,  related_name='context')
    book_id = IntegerField(null=False)
    cluster_id = ForeignKeyField(Cluster, related_name='cluster', null=True)
    value = TextField()
    start = IntegerField()
    end = IntegerField()
    type = TextField()

    def __str__(self):
        return f"{self.id}, {self.context_id}, {self.book_id}, {self.value}, {self.start}, {self.end}, {self.type}"

    class Meta:
        table_name = 'Ner'
