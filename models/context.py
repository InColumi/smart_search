from peewee import PrimaryKeyField, TextField, IntegerField
from models.base import BaseModel


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
