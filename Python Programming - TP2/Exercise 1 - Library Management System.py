# EXERCISE 1

# a
class Book:
    def __init__(self, title, author, isbn, available=True):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.available = available

# b
class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def remove_book(self, book):
        self.books.remove(book)

    def search_by_title(self, title):
        return [book for book in self.books if book.title.lower() == title.lower()]

    def search_by_author(self, author):
        return [book for book in self.books if book.author.lower() == author.lower()]

    def display_all_books(self):
        for book in self.books:
            print(f"\"{book.title}\" by {book.author} (ISBN: {book.isbn}) - Available: {book.available}")

    def borrow_book(self, book):
        if book.available:
            book.available = False
            print("Book borrowed successfully!")
        else:
            print("Sorry, this book is not available.")

    def return_book(self, book):
        if not book.available:
            book.available = True
            print("Book returned successfully!")
        else:
            print("This book is already available.")

library = Library()

# Add books
book1 = Book("Python Programming", "John Smith", "123")
book2 = Book("Data Science", "Jane Doe", "456", available=False)
library.add_book(book1)
library.add_book(book2)

# Display all books
library.display_all_books()

# Search for books
title = "Python Programming"
author = "Jane Doe"

books_by_title = library.search_by_title(title)
for book in books_by_title:
    print(f"Search Results for \"{title}\":")
    print(f"- {book.title} by {book.author} (ISBN: {book.isbn})")
books_by_author = library.search_by_author(author)
for book in books_by_author:
    print(f"Search Results for \"{author}\":")
    print(f"- {book.title} by {book.author} (ISBN: {book.isbn})")

# Borrow and return a book
library.borrow_book(book1)
library.return_book(book1)


'''[Running] python -u "h:\DAISI\Python Programming - TP2\Exercise 1 - Library Management System.py"
"Python Programming" by John Smith (ISBN: 123) - Available: True
"Data Science" by Jane Doe (ISBN: 456) - Available: False
Search Results for "Python Programming":
- Python Programming by John Smith (ISBN: 123)
Search Results for "Jane Doe":
- Data Science by Jane Doe (ISBN: 456)
Book borrowed successfully!
Book returned successfully!

[Done] exited with code=0 in 0.181 seconds'''