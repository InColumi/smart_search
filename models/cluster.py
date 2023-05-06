from peewee import PrimaryKeyField, TextField, IntegerField
from models.base import BaseModel


class Cluster(BaseModel):
    id = PrimaryKeyField()
    word = TextField()
    centroid = TextField()
    size = IntegerField()

    class Meta:
        table_name = 'Cluster'

    def __str__(self):
        return f"{self.id}, {self.word} {self.centroid}, {self.size}"
