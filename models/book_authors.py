from peewee import IntegerField
from models.base import BaseModel


class Book_authors(BaseModel):
    """Table with ner ifo"""
    ref_book_id = IntegerField(null=False)
    ref_authors_id = IntegerField(null=False)
   

    def __str__(self):
        return f"{self.ref_book_id}, {self.ref_authors_id}"

    class Meta:
        table_name = 'book_authors'
