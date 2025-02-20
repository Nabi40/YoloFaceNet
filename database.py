import psycopg2
from config import DB_CONFIG

def connect_db():
    """Establish connection to PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return None

def create_table():
    """Create the Attendance table with binary image storage."""
    conn = connect_db()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            create_script = '''
                CREATE TABLE IF NOT EXISTS Attendance (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(40) NOT NULL,
                    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accuracy INT,
                    image BYTEA  -- ✅ Store image as binary
                )
            '''
            cur.execute(create_script)
            conn.commit()
            print("✅ Attendance Table is ready!")
    except Exception as e:
        print(f"❌ Database Error: {e}")
    finally:
        if conn:
            conn.close()

def insert_attendance(name, accuracy, image_binary):
    """Insert a new attendance record with binary image into the database."""
    conn = connect_db()
    if not conn:
        print("❌ No database connection.")
        return

    try:
        with conn.cursor() as cur:
            insert_script = "INSERT INTO Attendance (name, time, accuracy, image) VALUES (%s, CURRENT_TIMESTAMP, %s, %s)"
            cur.execute(insert_script, (name, accuracy, psycopg2.Binary(image_binary)))
            conn.commit()
            print(f"✅ Attendance recorded: {name} with {accuracy}% accuracy and image saved in database.")
    except Exception as e:
        print(f"❌ Database Error: {e}")
    finally:
        if conn:
            conn.close()

def fetch_attendance():
    """Fetch the latest attendance records including binary images."""
    conn = connect_db()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, time, accuracy, image FROM Attendance ORDER BY time DESC")
            records = cur.fetchall()

            return [{"id": r[0], "name": r[1], "time": r[2].strftime("%Y-%m-%d %H:%M:%S"),
                     "accuracy": r[3], "image": r[4]} for r in records]
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Run only when executing this script directly
if __name__ == "__main__":
    create_table()
