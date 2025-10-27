import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', 5432)

def get_db_connection():
    """
    Establishes and returns a new database connection,
    setting the schema search path and session timezone to UTC.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT,
            # This is the updated line:
            options="-c search_path=derby,public -c timezone=UTC" 
        )
        return conn
    except Exception as e:
        print(f"Error: Could not connect to the database. {e}")
        return None

# A simple test function you can run
if __name__ == '__main__':
    conn = get_db_connection()
    if conn:
        print("Database connection successful!")
        conn.close()
    else:
        print("Database connection failed.")