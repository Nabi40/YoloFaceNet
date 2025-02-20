from database import insert_attendance

if __name__ == "__main__":
    # Example Data Entry
    name = input("Enter Name: ")
    accuracy = int(input("Enter Accuracy (%): "))

    # Read the image as binary data
    image_path = input("Enter image path (optional, press Enter to skip): ")
    image_binary = None
    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                image_binary = img_file.read()
            print("✅ Image file read successfully!")
        except FileNotFoundError:
            print("❌ Image file not found. Proceeding without image.")

    print("ℹ️ Sending data to insert_attendance...")
    insert_attendance(name, accuracy, image_binary if image_binary else b'')
