# Exercise 5

# b - i
# c - dictionary methods
# e - basic error hundling
def add_contact(phone_book):
    name = input("Enter contact name: ")
    number = input("Enter phone number: ")
    phone_book[name] = number
    print("Contact added successfully!")

# b - ii
def lookup_number(phone_book):
    name = input("Enter contact name to look up: ")
    if name in phone_book:
        print(f"Phone number for {name}: {phone_book[name]}")
    else:
        print("Contact not found!")

# b - iii
def delete_contact(phone_book):
    name = input("Enter contact name to delete: ")
    if name in phone_book:
        del phone_book[name]
        print("Contact deleted successfully!")
    else:
        print("Contact not found!")


# b - iv
def display_contacts(phone_book):
    if phone_book:
        print("\nAll Contacts:")
        for name, number in phone_book.items():
            print(f"{name}: {number}")
    else:
        print("Phone book is empty!")

def main():
    # a - empty dict to stor nums
    phone_book = {}
    while True:
        print("\nPhone Book Menu:")
        print("1. Add new contact")
        print("2. Look up a number")
        print("3. Delete a contact")
        print("4. Display all contacts")
        print("5. Quit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            # b - i
            add_contact(phone_book)
        elif choice == '2':
            # b - ii
            lookup_number(phone_book)
        elif choice == '3':
            # b - iii
            delete_contact(phone_book)
        elif choice == '4':
            # b - iv
            display_contacts(phone_book)
        elif choice == '5':
            # b - v
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()