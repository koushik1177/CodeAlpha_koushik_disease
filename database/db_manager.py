"""
Database Manager Module for Disease Prediction System.

Manages SQLite database connections, schema initialization,
and diagnostic history persistence for patient records
and risk prediction reports.
"""

import sqlite3
import json

from datetime import datetime
from typing import List, Dict, Any

from config.settings import DATABASE_PATH


class DatabaseManager:
    """
    SQLite Database Manager handling patient prediction logs
    and report persistence.
    """

    def __init__(self, db_path=DATABASE_PATH):
        """
        Initializes the database manager.

        Args:
            db_path: Path to the SQLite database.
        """

        self.db_path = str(db_path)

        # Make sure database directory exists
        database_dir = DATABASE_PATH.parent

        database_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self._init_db()


    # ========================================================
    # DATABASE CONNECTION
    # ========================================================

    def _get_connection(self) -> sqlite3.Connection:
        """
        Creates and returns a SQLite database connection.
        """

        conn = sqlite3.connect(
            self.db_path
        )

        conn.row_factory = sqlite3.Row

        return conn


    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def _init_db(self) -> None:
        """
        Initializes database schema if the table does not exist.
        """

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    disease_type TEXT NOT NULL,

                    disease_name TEXT NOT NULL,

                    patient_name TEXT
                        DEFAULT 'Anonymous Patient',

                    age INTEGER,

                    sex TEXT,

                    prediction INTEGER NOT NULL,

                    probability REAL NOT NULL,

                    risk_percentage REAL NOT NULL,

                    risk_category TEXT NOT NULL,

                    status TEXT NOT NULL,

                    timestamp DATETIME
                        DEFAULT CURRENT_TIMESTAMP,

                    input_data TEXT NOT NULL

                )
                """
            )

            conn.commit()


    # ========================================================
    # SAVE RECORD
    # ========================================================

    def save_record(
        self,
        disease_type: str,
        disease_name: str,
        patient_name: str,
        age: int,
        sex: str,
        prediction: int,
        probability: float,
        risk_percentage: float,
        risk_category: str,
        status: str,
        input_data: Dict[str, Any]
    ) -> int:
        """
        Saves a diagnostic prediction record to SQLite.

        Returns:
            int: ID of the inserted record.
        """

        sql = """
            INSERT INTO prediction_history (

                disease_type,
                disease_name,
                patient_name,
                age,
                sex,
                prediction,
                probability,
                risk_percentage,
                risk_category,
                status,
                timestamp,
                input_data

            )

            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """


        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        input_json = json.dumps(
            input_data
        )


        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                sql,
                (
                    disease_type,
                    disease_name,
                    patient_name,
                    age,
                    sex,
                    prediction,
                    probability,
                    risk_percentage,
                    risk_category,
                    status,
                    now,
                    input_json
                )
            )

            conn.commit()

            return cursor.lastrowid


    # ========================================================
    # GET RECENT PREDICTIONS
    # ========================================================

    def get_recent_predictions(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent prediction records.
        """

        if limit <= 0:
            limit = 50


        sql = """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """


        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                sql,
                (limit,)
            )

            rows = cursor.fetchall()


            return [
                dict(row)
                for row in rows
            ]


    # ========================================================
    # GET PREDICTION COUNT
    # ========================================================

    def get_prediction_count(self) -> int:
        """
        Returns the total number of prediction records.
        """

        sql = """
            SELECT COUNT(*) AS count
            FROM prediction_history
        """


        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                sql
            )

            row = cursor.fetchone()


            if row:

                return int(
                    row["count"]
                )

            return 0


    # ========================================================
    # SEARCH RECORDS
    # ========================================================

    def search_records(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Filters prediction history records matching
        a search term.
        """

        sql = """
            SELECT *
            FROM prediction_history

            WHERE
                patient_name LIKE ?
                OR disease_name LIKE ?
                OR status LIKE ?
                OR risk_category LIKE ?

            ORDER BY id DESC
        """


        term = f"%{query}%"


        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                sql,
                (
                    term,
                    term,
                    term,
                    term
                )
            )

            rows = cursor.fetchall()


            return [
                dict(row)
                for row in rows
            ]