from peewee import TextField, IntegerField, UUIDField
from models.base import BaseModel


class Titles(BaseModel):
    id = UUIDField()
    name = TextField()
    int_id = IntegerField()
    int_book_id = IntegerField()

    class Meta:
        table_name = 'titles'

    def __str__(self):
        return f"{self.id}, {self.name}, {self.int_id}, {self.int_book_id}"
