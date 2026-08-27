# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "73406ce0-1fcf-4b73-af3b-103e02a847e6",
# META       "default_lakehouse_name": "SharkData",
# META       "default_lakehouse_workspace_id": "0f8931b5-b1de-4408-8935-7082bb71d592",
# META       "known_lakehouses": [
# META         {
# META           "id": "73406ce0-1fcf-4b73-af3b-103e02a847e6"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Shark Attack Data — Ingest & Clean
# 
# Single notebook replacing **both** the pipeline and the Dataflow Gen2.
# 
# **Steps:**
# 1. Download `GSAF5.xls` from the web
# 2. Archive the raw XLS and a CSV copy to `Files/Raw/`
# 3. Load Excel data into `raw_sharkattacks` Delta table
# 4. Clean, parse dates, classify species
# 5. Write to `Shark_Attacks_Clean` Delta table
# 
# **Attach this notebook to the SharkData lakehouse before running.**

# CELL ********************

import re, os
from datetime import date

import requests
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType

# GSAF data contains ancient dates (pre-1582) - must set CORRECTED to avoid rebase error
spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1. Download GSAF5.xls

# CELL ********************

SOURCE_URL = "https://www.sharkattackfile.net/spreadsheets/GSAF5.xls"
SHEET_NAME = "Sheet1-GSAF"

RAW_FOLDER = "/lakehouse/default/Files/Raw"
XLS_PATH   = f"{RAW_FOLDER}/GSAF5.xls"
CSV_PATH   = f"{RAW_FOLDER}/GSAF5.csv"

os.makedirs(RAW_FOLDER, exist_ok=True)

print(f"Downloading {SOURCE_URL} ...")
resp = requests.get(SOURCE_URL, timeout=120)
resp.raise_for_status()

with open(XLS_PATH, "wb") as f:
    f.write(resp.content)

print(f"Saved XLS ({len(resp.content):,} bytes) to {XLS_PATH}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2. Read Excel → Pandas → archive CSV → Spark raw table

# CELL ********************

COLUMN_NAMES = [
    "Date", "Year", "Type", "Country", "State", "Location", "Activity",
    "Name", "Sex", "Age", "Injury", "Fatal", "Time", "Species",
    "Source", "pdf", "href_formula", "href", "Case_Number_1",
    "Case_Number_2", "Original_Order",
]

pdf_raw = pd.read_excel(
    XLS_PATH,
    sheet_name=SHEET_NAME,
    header=0,
    dtype=str,
)

pdf_raw = pdf_raw.iloc[:, :len(COLUMN_NAMES)]
pdf_raw.columns = COLUMN_NAMES

print(f"Rows read from Excel: {len(pdf_raw):,}")
print(f"Columns: {list(pdf_raw.columns)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

pdf_raw.to_csv(CSV_PATH, index=False)
print(f"CSV saved to {CSV_PATH}")

raw_df = spark.createDataFrame(pdf_raw.astype(str).fillna(""))
raw_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
    "Tables/raw_sharkattacks"
)

raw_count = spark.read.format("delta").load("Tables/raw_sharkattacks").count()
print(f"raw_sharkattacks written: {raw_count:,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3. Read raw table and filter junk rows

# CELL ********************

raw = spark.read.format("delta").load("Tables/raw_sharkattacks")

df = raw.filter(
    F.col("Year").isNotNull()
    & (F.trim(F.col("Year")) != "")
    & ~F.col("Year").isin("0", "0000", "5", "77", "500", '""')
)

print(f"After year filter: {df.count():,} rows (dropped {raw.count() - df.count():,} junk rows)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4. Robust date parsing
# 
# Handles all 5+ date formats in the GSAF data. Unparsed dates become `null` and are reported — never silently dropped.

# CELL ********************

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "nox": 11,
    "dec": 12, "december": 12,
}

_RE_ISO        = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RE_DD_MON_YY4 = re.compile(r"^(\d{1,2})[\s-]+([A-Za-z]+)[\s-]+(\d{4})$")
_RE_DD_MON_YY2 = re.compile(r"^(\d{1,2})-([A-Za-z]+)-(\d{2})$")
_RE_ORDINAL    = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)\s+([A-Za-z]+)\s*$")
_RE_MON_YY4    = re.compile(r"^([A-Za-z]{3,9})[.\s-]+(\d{4})$")
_RE_STRIP_PFX  = re.compile(r"^(?:early|late|mid|ca\.?|circa|around|approximately|reported)[\s-]+", re.IGNORECASE)
_RE_CASE_NUM   = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})")


def _safe_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def _parse_date(date_str, year_str, case_str):
    if not date_str:
        return None
    s = str(date_str).strip().replace("`", "").replace("Reported ", "")
    s = s.replace("--", "-")
    if not s:
        return None

    # ISO: 2025-01-11
    m = _RE_ISO.match(s)
    if m:
        return _safe_date(m.group(1), m.group(2), m.group(3))

    # DD Mon YYYY / DD-Mon-YYYY: "04 Mar 2024"
    m = _RE_DD_MON_YY4.match(s)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            return _safe_date(m.group(3), mon, m.group(1))

    # DD-Mon-YY (2-digit year)
    m = _RE_DD_MON_YY2.match(s)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        yr = int(m.group(3))
        yr = yr + 2000 if yr < 100 else yr
        if mon:
            return _safe_date(yr, mon, m.group(1))

    # Ordinal without year: "21st December" - borrow from Year column
    m = _RE_ORDINAL.match(s)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        yr = None
        if year_str:
            try:
                yr = int(str(year_str).strip())
            except (ValueError, TypeError):
                pass
        if mon and yr and yr > 0:
            return _safe_date(yr, mon, m.group(1))

    # Strip vague prefixes: "Early Nov-1966" -> "Nov-1966", "Ca. 1899" -> "1899"
    s2 = _RE_STRIP_PFX.sub("", s).strip()

    # Mon YYYY / Mon-YYYY - no specific day known, use day 1
    for candidate in ([s2] if s2 != s else []) + [s]:
        m = _RE_MON_YY4.match(candidate)
        if m:
            mon = _MONTHS.get(m.group(1).lower())
            if mon:
                return _safe_date(m.group(2), mon, 1)

    # Case number fallback: "1966.11.00" -> Nov 1 1966, "1940.00.00" -> skip (no month)
    if case_str:
        m = _RE_CASE_NUM.match(str(case_str).strip())
        if m:
            yr_c, mo_c, dy_c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mo_c > 0:
                return _safe_date(yr_c, mo_c, max(dy_c, 1))

    # Year-only fallback: "1940", "Ca. 1940", "Last incident of 1994 in Hong Kong"
    yr_hits = re.findall(r"\b(\d{4})\b", s)
    if yr_hits:
        try:
            yr = int(yr_hits[0])
            if 1000 <= yr <= 2100:
                return _safe_date(yr, 1, 1)
        except (ValueError, TypeError):
            pass

    return None


# UDF takes 3 args: date string, year column, case number
parse_date_udf = F.udf(lambda d, y, c: _parse_date(d, y, c), DateType())
print("Date parser registered.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = df.withColumn(
    "FinalDate",
    parse_date_udf(F.col("Date"), F.col("Year"), F.col("Case_Number_1")),
)

total  = df.count()
parsed = df.filter(F.col("FinalDate").isNotNull()).count()
print(f"Total rows: {total:,}  |  Dates resolved: {parsed:,}  |  Still null: {total - parsed:,}")
print("(Null dates are kept in output - day/month unknown even from case number)")

if total - parsed > 0:
    print("\nRemaining unparsed:")
    df.filter(F.col("FinalDate").isNull()).select("Date", "Year", "Case_Number_1").show(20, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 5. Species classification

# CELL ********************

_SPECIES_MAP = [
    ("sand tiger", "Sand Tiger"), ("whitetip reef", "Whitetip Reef"),
    ("oceanic", "Oceanic White Tip"), ("white tip", "White Tip"),
    ("whitetip", "White Tip"), ("white", "Great White"),
    ("zambezi", "Bull"), ("zambesi", "Bull"), ("leucas", "Bull"), ("bull", "Bull"),
    ("blue pointer", "Mako"), ("bonita", "Mako"), ("bonito", "Mako"), ("mako", "Mako"),
    ("ragged", "Sand Tiger"), ("grey nurse", "Sand Tiger"), ("tiger", "Tiger"),
    ("hammerhead", "Hammerhead"), ("galapagos", "Galapagos"),
    ("7-gill", "Seven Gill"), ("sevengill", "Seven Gill"),
    ("blue", "Blue"), ("dog", "Dog"), ("wobbegong", "Wobbegong"),
    ("blacktip", "Blacktip"), ("silky", "Silky"), ("shovelnose", "Shovelnose"),
    ("basking", "Basking"), ("nurse", "Nurse"), ("angel", "Angel"),
    ("marcrurus", "Dusky"), ("dusky", "Dusky"), ("whaler", "Whaler"),
    ("thresher", "Thresher"), ("lemon", "Lemon"), ("porbeagle", "Porbeagle"),
    ("carpet", "Carpet"), ("sand", "Sand Tiger"), ("leopard", "Leopard"),
    ("reef", "Grey Reef"), ("gummy", "Gummy"), ("horn", "Horn"),
    ("copper", "Copper"), ("black", "Black Tip"), ("spinner", "Spinner"),
    ("salmon", "Salmon"),
]


def _classify_species(shark_type):
    if not shark_type:
        return "Unknown"
    s = shark_type.strip().lower()
    for pattern, species in _SPECIES_MAP:
        if pattern in s:
            return species
    return "Unknown"


classify_udf = F.udf(_classify_species, StringType())
print(f"Species classifier: {len(_SPECIES_MAP)} patterns")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 6. Build the clean table

# CELL ********************

clean = df.filter(~F.upper(F.trim(F.col("Type"))).isin("INVALID"))

clean = (
    clean
    .withColumnRenamed("Species", "Shark_Type_Raw")
    .withColumnRenamed("Original_Order", "Report_Order")
    .withColumnRenamed("href", "Report")
)

clean = (
    clean
    .withColumn("Date", F.col("FinalDate"))
    .withColumn("Year", F.year(F.col("FinalDate")).cast(StringType()))
    .withColumn("Month_Name", F.date_format(F.col("FinalDate"), "MMMM"))
    .withColumn("Month_Number", F.date_format(F.col("FinalDate"), "MM"))
    .withColumn("Day", F.date_format(F.col("FinalDate"), "dd"))
)

clean = (
    clean
    .withColumn("Type", F.regexp_replace(F.col("Type"), "`", ""))
    .withColumn("Fatal", F.trim(F.col("Fatal")))
    .withColumn("Fatal",
        F.when(F.col("Fatal").isin("", None), "N")
         .when(F.col("Fatal").contains("F"), "Y")
         .otherwise(F.col("Fatal")))
    .withColumn("Sex", F.trim(F.regexp_replace(F.col("Sex"), "\\.", "")))
    .withColumn("Sex",
        F.when(F.col("Sex") == "N", "M")
         .when(F.col("Sex") == "lli", "M")
         .when(F.col("Sex").isin("", None), "Unknown")
         .otherwise(F.col("Sex")))
    .withColumn("Report", F.regexp_replace(F.coalesce(F.col("Report"), F.lit("")), "`", ""))
    .withColumn("Shark_Type", F.lower(F.coalesce(F.col("Shark_Type_Raw"), F.lit(""))))
    .withColumn("Case_Number", F.coalesce(F.col("Case_Number_1"), F.lit("")))
)

clean = clean.withColumn("Species", classify_udf(F.col("Shark_Type")))

clean = (
    clean
    .withColumn("State", F.coalesce(F.col("State"), F.lit("")))
    .withColumn("Country", F.coalesce(F.col("Country"), F.lit("")))
    .withColumn("Location", F.coalesce(F.col("Location"), F.lit("")))
    .withColumn("Region",
        F.when(F.trim(F.col("State")) == "", F.col("Country"))
         .otherwise(F.concat_ws(", ", F.col("State"), F.col("Country"))))
    .withColumn("Place",
        F.concat_ws(", ",
            F.when(F.trim(F.col("Location")) != "", F.col("Location")),
            F.when(F.trim(F.col("State")) != "", F.col("State")),
            F.col("Country")))
)

print(f"Clean rows: {clean.count():,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 7. Validate and write

# CELL ********************

output = clean.select(
    "Date", "Year", "Month_Name", "Month_Number", "Day",
    "Case_Number", "State", "Type", "Country", "Location",
    "Activity", "Name", "Sex", "Age", "Time",
    "Injury", "Source", "pdf", "Report", "Fatal",
    "Report_Order", "Shark_Type", "Species", "Region", "Place",
).orderBy(F.col("Date").desc())

print("Year distribution (recent):")
output.groupBy("Year").count().orderBy(F.desc("Year")).show(10)

print(f"\nTotal output rows: {output.count():,}")
output.show(5, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

output.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
    "Tables/Shark_Attacks_Clean"
)

final_count = spark.read.format("delta").load("Tables/Shark_Attacks_Clean").count()
print(f"Shark_Attacks_Clean written: {final_count:,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 8. Write SharkAttacks_Silver
# 
# Builds the Silver table consumed by the Attack Report semantic model.
# Schema matches the original `fabric_shark_analysis` notebook exactly.

# CELL ********************

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, TimestampType
)
from pyspark.sql import functions as F

silver_schema = StructType([
    StructField("date",           TimestampType(), True),
    StructField("year",           IntegerType(),   True),
    StructField("type",           StringType(),    True),
    StructField("country",        StringType(),    True),
    StructField("state",          StringType(),    True),
    StructField("location",       StringType(),    True),
    StructField("activity",       StringType(),    True),
    StructField("name",           StringType(),    True),
    StructField("sex",            StringType(),    True),
    StructField("age",            IntegerType(),   True),
    StructField("injury",         StringType(),    True),
    StructField("fatal",          StringType(),    True),
    StructField("time",           StringType(),    True),
    StructField("species",        StringType(),    True),
    StructField("source",         StringType(),    True),
    StructField("pdf",            StringType(),    True),
    StructField("href_formula",   StringType(),    True),
    StructField("href",           StringType(),    True),
    StructField("case_number",    StringType(),    True),
    StructField("case_number_1",  StringType(),    True),
    StructField("original_order", IntegerType(),   True),
])

# Shark_Attacks_Clean columns: Date, Year, Month_Name, Month_Number, Day,
#   Case_Number, State, Type, Country, Location, Activity, Name, Sex, Age,
#   Time, Injury, Source, pdf, Report, Fatal, Report_Order, Shark_Type,
#   Species, Region, Place
# href_formula was not carried through the clean pipeline - set to null
# Case_Number_1 was merged into Case_Number - set case_number_1 to null
clean_src = spark.read.format("delta").load("Tables/Shark_Attacks_Clean")

silver = (
    clean_src
    .withColumn("date",           F.col("Date").cast(TimestampType()))
    .withColumn("year",           F.col("Year").cast(IntegerType()))
    .withColumn("type",           F.col("Type"))
    .withColumn("country",        F.col("Country"))
    .withColumn("state",          F.col("State"))
    .withColumn("location",       F.col("Location"))
    .withColumn("activity",       F.col("Activity"))
    .withColumn("name",           F.col("Name"))
    .withColumn("sex",            F.col("Sex"))
    .withColumn("age",            F.col("Age").cast(IntegerType()))
    .withColumn("injury",         F.col("Injury"))
    .withColumn("fatal",          F.col("Fatal"))
    .withColumn("time",           F.col("Time"))
    .withColumn("species",        F.col("Species"))
    .withColumn("source",         F.col("Source"))
    .withColumn("pdf",            F.col("pdf"))
    .withColumn("href_formula",   F.lit(None).cast(StringType()))
    .withColumn("href",           F.col("Report"))
    .withColumn("case_number",    F.col("Case_Number"))
    .withColumn("case_number_1",  F.lit(None).cast(StringType()))
    .withColumn("original_order", F.col("Report_Order").cast(IntegerType()))
    .select([f.name for f in silver_schema])
)

# Replace "nan" strings with null
for field in silver_schema:
    if isinstance(field.dataType, StringType):
        silver = silver.withColumn(field.name,
            F.when(F.col(field.name) == "nan", None).otherwise(F.col(field.name)))

silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
    "Tables/SharkAttacks_Silver"
)

count = spark.read.format("delta").load("Tables/SharkAttacks_Silver").count()
print(f"SharkAttacks_Silver written: {count:,} rows")
silver.groupBy("year").count().orderBy(F.desc("year")).show(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
