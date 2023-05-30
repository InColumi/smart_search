from peewee import PrimaryKeyField, TextField
from models.base import BaseModel


class Centroid(BaseModel):
    id = PrimaryKeyField()
    value = TextField()
   
    class Meta:
        table_name = 'centroid'

    def __str__(self):
        return f"{self.id}, {self.value}"
