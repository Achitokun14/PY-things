#Exercise 1
import csv
import json
from typing import List, Dict
import pandas as pd
from pathlib import Path


# 2
class StudentOperationsSystem:
    def __init__(self, students: List[Dict]):
        self.students = students
        self.csv_file = 'students.csv'
        self.high_performers_file = 'high_performers.csv'
        self.json_file = 'students.json'

    def save_to_csv(self, filename: str, data: List[Dict]) -> None:
        """Save student records to a CSV file."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'grade', 'courses']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for student in data:
                    student_copy = student.copy()
                    student_copy['courses'] = ','.join(student_copy['courses'])
                    writer.writerow(student_copy)
            print(f"Successfully saved data to {filename}")
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")

    def read_csv_and_add_student(self, new_student: Dict) -> None:
        """Read CSV file and add a new student."""
        try:
            if Path(self.csv_file).exists():
                df = pd.read_csv(self.csv_file)
                df['courses'] = df['courses'].str.split(',')
                
                self.students = df.to_dict('records')
            
            self.students.append(new_student)
            self.save_to_csv(self.csv_file, self.students)
            print(f"Successfully added new student: {new_student['name']}")
        except Exception as e:
            print(f"Error reading CSV and adding student: {str(e)}")

    def calculate_average_grade(self) -> float:
        """Calculate and return the average grade of all students."""
        try:
            grades = [student['grade'] for student in self.students]
            average = sum(grades) / len(grades)
            return round(average, 2)
        except ZeroDivisionError:
            print("No students in the system")
            return 0
        except Exception as e:
            print(f"Error calculating average grade: {str(e)}")
            return 0

    def export_high_performers(self, min_grade: int = 80) -> None:
        """Export students with grades above the specified threshold."""
        try:
            high_performers = [student for student in self.students 
                             if student['grade'] > min_grade]
            self.save_to_csv(self.high_performers_file, high_performers)
            print(f"Successfully exported {len(high_performers)} high performers")
        except Exception as e:
            print(f"Error exporting high performers: {str(e)}")

    def save_to_json(self) -> None:
        """Save student records to JSON with proper indentation."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(self.students, jsonfile, indent=4, ensure_ascii=False)
            print(f"Successfully saved data to {self.json_file}")
        except Exception as e:
            print(f"Error saving to JSON: {str(e)}")

def main():
    # 1
    students = [
        {'id': '001', 'name': 'John Doe', 'grade': 85, 'courses': ['Math', 'Physics']},
        {'id': '002', 'name': 'Jane Smith', 'grade': 92, 'courses': ['Chemistry', 'Biology']},
        {'id': '003', 'name': 'Bob Wilson', 'grade': 78, 'courses': ['Physics', 'Math']}
    ]

    sms = StudentOperationsSystem(students)

    sms.save_to_csv(sms.csv_file, students)

    new_student = {
        'id': '004',
        'name': 'Alice Johnson',
        'grade': 88,
        'courses': ['English', 'History']
    }
    sms.read_csv_and_add_student(new_student)

    avg_grade = sms.calculate_average_grade()
    print(f"Average grade: {avg_grade}")

    sms.export_high_performers()

    sms.save_to_json()

if __name__ == "__main__":
    main()


'''
results in those files :
students.csv
students.json
high_performers.csv

and also terminal:
Successfully saved data to students.csv
Successfully saved data to students.csv
Successfully added new student: Alice Johnson
Average grade: 85.75
Successfully saved data to high_performers.csv
Successfully exported 3 high performers
Successfully saved data to students.json

I refined that code with AI to look perfect
'''