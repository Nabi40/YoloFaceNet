from database import insert_attendance

if __name__ == "__main__":
    # Example Data Entry
    name = input("Enter Name: ")
    accuracy = int(input("Enter Accuracy (%): "))

    insert_attendance(name, accuracy)
