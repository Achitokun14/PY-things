# Exercise 3
import numpy as np

# a
num_students = int(input("Enter the number of students: "))
num_subjects = int(input("Enter the number of subjects: "))

student_marks = np.random.randint(0, 21, size=(num_students, num_subjects))

# b
total_marks = np.sum(student_marks, axis=1)
percentages = (total_marks / (num_subjects * 20)) * 100
subject_averages = np.mean(student_marks, axis=0)
grades = np.where(percentages >= 16, 'A',np.where(percentages >= 14, 'B',np.where(percentages >= 12, 'C',np.where(percentages >= 10, 'D', 'F'))))

print("Student Results:")
for i in range(num_students):
    print(f"Student {i+1}: Total={total_marks[i]}/{num_subjects*20} ({percentages[i]:.2f}%) Grade: {grades[i]}")

print("\nSubject Averages:")
for i in range(num_subjects):
    print(f"Subject {i+1}: {subject_averages[i]:.1f}")



'''
Enter the number of students: 1
Enter the number of subjects: 4
Student Results:
Student 1: Total=44/80 (55.00%) Grade: A

Subject Averages:
Subject 1: 0.0
Subject 2: 9.0
Subject 3: 19.0
Subject 4: 16.0
'''