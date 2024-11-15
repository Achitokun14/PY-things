#Exercise 2
import sqlite3
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

#additional helping tools ...
@dataclass
class Book:
    title: str
    author: str
    year: int
    available: bool = True
    book_id: Optional[int] = None

class LibraryDatabase:
    def __init__(self, db_name: str = "library.db"):
        self.db_name = db_name
        self.create_database()

    def create_database(self) -> None:
        """ 1 -- Create the database and books table if they don't exist."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS books (
                        book_id INTEGER PRIMARY KEY,
                        title TEXT NOT NULL,
                        author TEXT NOT NULL,
                        year INTEGER,
                        available BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_author ON books(author)
                ''')
                
                conn.commit()
                print("Database and table created successfully")
        except sqlite3.Error as e:
            print(f"Error creating database: {str(e)}")

    def add_book(self, book: Book) -> bool:
        """ 2 -- Add a new book to the database."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO books (title, author, year, available)
                    VALUES (?, ?, ?, ?)
                ''', (book.title, book.author, book.year, book.available))
                conn.commit()
                print(f"Successfully added book: {book.title}")
                return True
        except sqlite3.Error as e:
            print(f"Error adding book: {str(e)}")
            return False

    def list_available_books(self) -> List[Book]:
        """ 3 -- Show all available books."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT book_id, title, author, year, available
                    FROM books
                    WHERE available = TRUE
                    ORDER BY title
                ''')
                books = [Book(*row) for row in cursor.fetchall()]
                return books
        except sqlite3.Error as e:
            print(f"Error listing available books: {str(e)}")
            return []

    def find_books_by_author(self, author: str) -> List[Book]:
        """ 4 -- List all books by a specific author."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT book_id, title, author, year, available
                    FROM books
                    WHERE author LIKE ?
                    ORDER BY year DESC
                ''', (f'%{author}%',))
                books = [Book(*row) for row in cursor.fetchall()]
                return books
        except sqlite3.Error as e:
            print(f"Error finding books by author: {str(e)}")
            return []

    def export_to_dataframe(self, books: List[Book]) -> pd.DataFrame:
        """Convert books list to pandas DataFrame."""
        return pd.DataFrame([vars(book) for book in books])

def main():
    library_db = LibraryDatabase()

    sample_books = [
        Book("The Great Gatsby", "F. Scott Fitzgerald", 1925),
        Book("To Kill a Mockingbird", "Harper Lee", 1960),
        Book("1984", "George Orwell", 1949),
        Book("Pride and Prejudice", "Jane Austen", 1813),
        Book("The Catcher in the Rye", "J.D. Salinger", 1951)
    ]

    print("\nAdding sample books:")
    for book in sample_books:
        library_db.add_book(book)

    print("\nAvailable books:")
    available_books = library_db.list_available_books()
    for book in available_books:
        print(f"- {book.title} by {book.author} ({book.year})")

    author_search = "Orwell"
    print(f"\nSearching for books by {author_search}:")
    author_books = library_db.find_books_by_author(author_search)
    for book in author_books:
        print(f"- {book.title} ({book.year})")

    print("\nExporting results to DataFrame:")
    df = library_db.export_to_dataframe(library_db.list_available_books())
    print("\nDataFrame contents:")
    print(df.to_string(index=False))

    csv_filename = "library_books.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nExported data to {csv_filename}")


if __name__ == "__main__":
    main()


'''
results in those files :
library.db (open it in a db visualiser like dbeaver)
library_books.csv

and also terminal:
Database and table created successfully

Adding sample books:
Successfully added book: The Great Gatsby
Successfully added book: To Kill a Mockingbird
Successfully added book: 1984
Successfully added book: Pride and Prejudice
Successfully added book: The Catcher in the Rye

Available books:
- 3 by 1984 (George Orwell)
- 4 by Pride and Prejudice (Jane Austen)
- 5 by The Catcher in the Rye (J.D. Salinger)
- 1 by The Great Gatsby (F. Scott Fitzgerald)
- 2 by To Kill a Mockingbird (Harper Lee)

Searching for books by Orwell:
- 3 (George Orwell)

Exporting results to DataFrame:

DataFrame contents:
 title                 author                year  available  book_id
     3                   1984       George Orwell       1949        1
     4    Pride and Prejudice         Jane Austen       1813        1
     5 The Catcher in the Rye       J.D. Salinger       1951        1
     1       The Great Gatsby F. Scott Fitzgerald       1925        1
     2  To Kill a Mockingbird          Harper Lee       1960        1

Exported data to library_books.csv



I refined that code with AI to look perfect
'''