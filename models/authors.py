from peewee import TextField, IntegerField, UUIDField
from models.base import BaseModel


class Author(BaseModel):
    id = UUIDField()
    name = TextField()
    int_id =IntegerField()


    class Meta:
        table_name = 'authors'

    def __str__(self):
        return f"{self.id}, {self.name}, {self.int_id}"
