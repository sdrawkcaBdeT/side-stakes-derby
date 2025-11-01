import os
import sys
from derby_game.database.connection import get_db_connection

def initialize_database():
    """
    Reads the schema.sql file and executes it to create the database tables.
    This is a RESET script: it will DROP the 'derby' schema if it exists
    and create it fresh from the schema.sql file.
    """
    try:
        with open('derby_game/database/schema.sql', 'r') as f:
            # This should contain the "CREATE SCHEMA IF NOT EXISTS derby;" line
            sql_commands = f.read()
    except FileNotFoundError:
        print("Error: schema.sql not found. Make sure it's in derby_game/database/")
        return

    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("Failed to get database connection.")
        
        # --- Step 1: Drop the old schema ---
        # We must set autocommit=True to run DROP SCHEMA
        # because it can't run inside a transaction block.
        conn.autocommit = True 
        with conn.cursor() as cur:
            print("Dropping existing 'derby' schema (if it exists)...")
            cur.execute("DROP SCHEMA IF EXISTS derby CASCADE;")
            print("'derby' schema dropped.")
        
        # Turn autocommit back off to run the rest as a transaction
        conn.autocommit = False

        # --- Step 2: Create and populate the new schema ---
        with conn.cursor() as cur:
            
            # --- Create Tables ---
            print("Creating new 'derby' schema and tables...")
            cur.execute(sql_commands) # This runs your whole schema.sql file
            print("Database tables created successfully!")
            
            # --- PRE-POPULATE BOTS ---
            print("Populating bot trainers...")
            cur.execute("""
                INSERT INTO derby.trainers (user_id, is_bot)
                VALUES 
                    (1, TRUE), -- PaddockPete
                    (2, TRUE), -- StartingGateSally
                    (3, TRUE)  -- FirstTurnFrank
                ON CONFLICT (user_id) DO NOTHING; 
            """)
            print(f"Added/updated {cur.rowcount} bot trainers.")
        
        # Once the 'with' block is done, commit the transaction
        conn.commit()
        print("All changes committed to the database.")

    except Exception as e:
        if conn:
            # Rollback any pending transaction if an error occurred
            conn.rollback() 
            print("An error occurred. Transaction rolled back.")
        print(f"Error details: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    print("This script will RESET your 'derby' database schema.")
    print("WARNING: All existing data in the 'derby' schema will be WIPED.")
    response = input("Are you sure you want to continue? (y/n): ")
    
    if response.lower() == 'y':
        initialize_database()
    else:
        print("Database initialization cancelled.")
