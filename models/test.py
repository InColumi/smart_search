from peewee import PrimaryKeyField, TextField, IntegerField, ForeignKeyField
from models.centroid import Centroid
from models.base import BaseModel


class Test(BaseModel):
    id = PrimaryKeyField()
    context = TextField()
    id_book = IntegerField()
    centroid_id = ForeignKeyField(Centroid,  related_name='centroid')

    class Meta:
        table_name = 'test'

    def __str__(self):
        return f"{self.id}, {self.context} {self.id_book}, {self.centroid_id}"
