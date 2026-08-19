"""
Database Inspector & Diagnostic Prediction History Module.

Allows users to inspect the SQLite database schema, raw tables,
prediction counts, search patient logs, and export diagnostic records.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from database.db_manager import DatabaseManager
from config.settings import DATABASE_PATH

from services.report_service import (
    generate_pdf_report,
    generate_csv_report,
    generate_txt_report
)


# ============================================================
# HISTORY PAGE
# ============================================================

def render_history_page():
    """
    Renders the SQLite database inspector and prediction history page.
    """

    st.markdown(
        "### 🗄️ SQLite Database Inspector & History Logs"
    )

    st.write(
        "Inspect SQLite prediction records, search patient logs, "
        "and re-export diagnostic reports."
    )


    # ========================================================
    # DATABASE
    # ========================================================

    db = DatabaseManager()


    # ========================================================
    # DATABASE METADATA
    # ========================================================

    db_file = Path(
        DATABASE_PATH
    )


    file_size_kb = (
        round(
            os.path.getsize(db_file) / 1024,
            2
        )
        if db_file.exists()
        else 0.0
    )


    total_count = (
        db.get_prediction_count()
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Database File",
            db_file.name
        )


    with c2:

        st.metric(
            "Total DB Records",
            total_count
        )


    with c3:

        st.metric(
            "Database Size",
            f"{file_size_kb} KB"
        )


    with c4:

        st.metric(
            "Storage Status",
            "Active (SQLite3)"
        )


    st.markdown("---")


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2 = st.tabs(
        [
            "📜 Patient Diagnostic Logs",
            "🔍 Raw SQLite Table Inspector"
        ]
    )


    # ========================================================
    # TAB 1 - PATIENT LOGS
    # ========================================================

    with tab1:

        search_query = st.text_input(

            "🔍 Search Database Records "
            "by Patient Name, Disease, or Risk Category:",

            placeholder=(
                "e.g. John Doe, "
                "Heart Disease, High Risk"
            ),

            key="db_search_input"
        )


        # ----------------------------------------------------
        # SEARCH / RECENT RECORDS
        # ----------------------------------------------------

        if search_query.strip():

            records = db.search_records(
                search_query.strip()
            )

        else:

            records = db.get_recent_predictions(
                limit=50
            )


        # ----------------------------------------------------
        # NO RECORDS
        # ----------------------------------------------------

        if not records:

            st.info(
                "No matching database records found."
            )


        else:

            df_history = pd.DataFrame(
                records
            )


            # ------------------------------------------------
            # RECORD TABLE
            # ------------------------------------------------

            st.markdown(
                "#### 📄 Database Records Table"
            )


            display_columns = [
                "id",
                "patient_name",
                "disease_name",
                "risk_percentage",
                "risk_category",
                "status",
                "timestamp"
            ]


            display_df = (
                df_history[
                    display_columns
                ]
                .rename(
                    columns={
                        "id": "ID",

                        "patient_name":
                            "Patient Name",

                        "disease_name":
                            "Disease Evaluated",

                        "risk_percentage":
                            "Risk %",

                        "risk_category":
                            "Severity",

                        "status":
                            "Diagnostic Status",

                        "timestamp":
                            "Record Timestamp"
                    }
                )
            )


            st.dataframe(
                display_df,
                use_container_width=True
            )


            # ------------------------------------------------
            # EXPORT CSV
            # ------------------------------------------------

            st.markdown(
                "#### 📥 Export Records"
            )


            csv_full = (
                df_history.to_csv(
                    index=False
                )
            )


            st.download_button(

                "📥 Export Filtered DB Records (CSV)",

                data=csv_full,

                file_name=(
                    "SQLite_Medical_Records_Export.csv"
                ),

                mime="text/csv",

                type="primary"
            )


            st.markdown("---")


            # ------------------------------------------------
            # RE-GENERATE REPORT
            # ------------------------------------------------

            st.markdown(
                "#### 📄 Re-Generate Patient Diagnostic Report"
            )


            record_ids = [
                record["id"]
                for record in records
            ]


            selected_id = st.selectbox(

                "Select Patient Record ID:",

                options=record_ids,

                key="selected_patient_record"
            )


            selected_record = next(

                (
                    record
                    for record in records
                    if record["id"] == selected_id
                ),

                None
            )


            if selected_record:

                result_dict = {

                    "disease":
                        selected_record[
                            "disease_name"
                        ],

                    "prediction":
                        selected_record[
                            "prediction"
                        ],

                    "probability":
                        selected_record[
                            "probability"
                        ],

                    "risk_percentage":
                        selected_record[
                            "risk_percentage"
                        ],

                    "risk_category":
                        selected_record[
                            "risk_category"
                        ],

                    "status":
                        selected_record[
                            "status"
                        ]
                }


                col1, col2, col3 = st.columns(3)


                # ------------------------------------------------
                # PDF
                # ------------------------------------------------

                with col1:

                    pdf_bytes = (
                        generate_pdf_report(
                            result_dict,
                            selected_record[
                                "patient_name"
                            ]
                        )
                    )


                    st.download_button(

                        "📄 Download PDF Report",

                        data=pdf_bytes,

                        file_name=(
                            f"Patient_Report_"
                            f"{selected_id}.pdf"
                        ),

                        mime="application/pdf",

                        use_container_width=True
                    )


                # ------------------------------------------------
                # CSV
                # ------------------------------------------------

                with col2:

                    csv_str = (
                        generate_csv_report(
                            result_dict,
                            selected_record[
                                "patient_name"
                            ]
                        )
                    )


                    st.download_button(

                        "📊 Download CSV Record",

                        data=csv_str,

                        file_name=(
                            f"Patient_Record_"
                            f"{selected_id}.csv"
                        ),

                        mime="text/csv",

                        use_container_width=True
                    )


                # ------------------------------------------------
                # TXT
                # ------------------------------------------------

                with col3:

                    txt_str = (
                        generate_txt_report(
                            result_dict,
                            selected_record[
                                "patient_name"
                            ]
                        )
                    )


                    st.download_button(

                        "📝 Download TXT Summary",

                        data=txt_str,

                        file_name=(
                            f"Patient_Summary_"
                            f"{selected_id}.txt"
                        ),

                        mime="text/plain",

                        use_container_width=True
                    )


    # ========================================================
    # TAB 2 - RAW SQLITE INSPECTOR
    # ========================================================

    with tab2:

        st.markdown(
            "#### 🛠️ Raw SQLite Schema & Columns"
        )


        st.code(
            """
CREATE TABLE prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_type TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    patient_name TEXT DEFAULT 'Anonymous Patient',
    age INTEGER,
    sex TEXT,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    risk_percentage REAL NOT NULL,
    risk_category TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_data TEXT NOT NULL
);
            """,
            language="sql"
        )


        # ----------------------------------------------------
        # RAW RECORDS
        # ----------------------------------------------------

        records_all = (
            db.get_recent_predictions(
                limit=100
            )
        )


        if records_all:

            st.markdown(
                "#### 🔍 Raw JSON Input Features Payload"
            )


            df_raw = pd.DataFrame(
                records_all
            )


            st.dataframe(
                df_raw,
                use_container_width=True
            )

        else:

            st.info(
                "No database records available."
            )