"""
Fetch medicine labels, convert them to documents, and split them into chunks.
"""
import os
from typing import Any, Dict, List
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.faiss_store import save_vector_store


load_dotenv()


OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# For version 1, keep this list controlled.
# We can later expand to 50, 100, or full openFDA dataset ingestion.
MEDICINES = [
    {
        "display_name": "Paracetamol / Acetaminophen",
        "openfda_query_name": "acetaminophen",
        "aliases": ["paracetamol", "acetaminophen", "dolo 650", "crocin", "calpol", "tylenol"],
    },
    {
        "display_name": "Ibuprofen",
        "openfda_query_name": "ibuprofen",
        "aliases": ["ibuprofen", "advil", "motrin"],
    },
    {
        "display_name": "Aspirin",
        "openfda_query_name": "aspirin",
        "aliases": ["aspirin"],
    },
    {
        "display_name": "Cetirizine",
        "openfda_query_name": "cetirizine",
        "aliases": ["cetirizine", "zyrtec"],
    },
    {
        "display_name": "Loratadine",
        "openfda_query_name": "loratadine",
        "aliases": ["loratadine", "claritin"],
    },
    {
        "display_name": "Metformin",
        "openfda_query_name": "metformin",
        "aliases": ["metformin"],
    },
    {
        "display_name": "Amoxicillin",
        "openfda_query_name": "amoxicillin",
        "aliases": ["amoxicillin"],
    },
    {
        "display_name": "Azithromycin",
        "openfda_query_name": "azithromycin",
        "aliases": ["azithromycin", "zithromax"],
    },
    {
        "display_name": "Omeprazole",
        "openfda_query_name": "omeprazole",
        "aliases": ["omeprazole", "prilosec"],
    },
    {
        "display_name": "Atorvastatin",
        "openfda_query_name": "atorvastatin",
        "aliases": ["atorvastatin", "lipitor"],
    },
]


LABEL_FIELDS = {
    "purpose": "Purpose",
    "indications_and_usage": "Uses / Indications",
    "warnings": "Warnings",
    "boxed_warning": "Boxed Warning",
    "adverse_reactions": "Side Effects / Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "contraindications": "Contraindications",
    "precautions": "Precautions",
    "dosage_and_administration": "Dosage Information",
    "pregnancy": "Pregnancy Information",
    "nursing_mothers": "Nursing Mothers Information",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "overdosage": "Overdosage Information",
    "patient_medication_information": "Patient Medication Information",
}


def to_text(value: Any) -> str:
    """
    openFDA fields are often list[str].
    This converts them into a clean string.
    """
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())

    if isinstance(value, str):
        return value.strip()

    return ""


def fetch_medicine_records(openfda_query_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch medicine label records from openFDA by generic name.

    openFDA API key is optional for small usage.
    If you have one later, add OPENFDA_API_KEY to .env.
    """

    search_query = f'openfda.generic_name:"{openfda_query_name}"'
    encoded_query = quote(search_query)

    url = f"{OPENFDA_LABEL_URL}?search={encoded_query}&limit={limit}"

    openfda_api_key = os.getenv("OPENFDA_API_KEY")
    if openfda_api_key:
        url += f"&api_key={openfda_api_key}"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 404:
            print(f"[WARN] No records found for: {openfda_query_name}")
            return []

        response.raise_for_status()

        data = response.json()
        return data.get("results", [])

    except requests.RequestException as exc:
        print(f"[ERROR] Failed to fetch {openfda_query_name}: {exc}")
        return []


def convert_record_to_plain_text(
    record: Dict[str, Any],
    display_name: str,
    query_name: str,
    aliases: List[str],
) -> str:
    """
    Converts an openFDA label record into readable plain text.
    This text will be chunked and embedded.
    """

    openfda = record.get("openfda", {})

    generic_name = ", ".join(openfda.get("generic_name", []))
    brand_name = ", ".join(openfda.get("brand_name", []))
    manufacturer = ", ".join(openfda.get("manufacturer_name", []))

    text_parts = [
        f"Medicine name: {display_name}",
        f"openFDA query name: {query_name}",
        f"Common aliases or brand references: {', '.join(aliases)}",
        "",
        f"Generic name: {generic_name or 'Not available'}",
        f"Brand name: {brand_name or 'Not available'}",
        f"Manufacturer: {manufacturer or 'Not available'}",
        "",
    ]

    for field_key, field_title in LABEL_FIELDS.items():
        field_text = to_text(record.get(field_key))

        if field_text:
            text_parts.append(f"{field_title}:")
            text_parts.append(field_text)
            text_parts.append("")

    text_parts.append("Source: openFDA Drug Label API.")
    text_parts.append(
        "This information is for general educational purposes only and should not replace medical advice from a doctor or pharmacist."
    )

    return "\n".join(text_parts).strip()


def create_document_from_record(
    record: Dict[str, Any],
    medicine: Dict[str, Any],
) -> Document:
    """
    Creates one LangChain Document from one openFDA record.
    Metadata will be copied to every chunk after splitting.
    """

    display_name = medicine["display_name"]
    query_name = medicine["openfda_query_name"]
    aliases = medicine["aliases"]

    page_content = convert_record_to_plain_text(
        record=record,
        display_name=display_name,
        query_name=query_name,
        aliases=aliases,
    )

    openfda = record.get("openfda", {})

    metadata = {
        "medicine_name": display_name,
        "openfda_query_name": query_name,
        "aliases": aliases,
        "generic_names": openfda.get("generic_name", []),
        "brand_names": openfda.get("brand_name", []),
        "manufacturer_names": openfda.get("manufacturer_name", []),
        "source": "openFDA Drug Label API",
        "source_url": OPENFDA_LABEL_URL,
        "set_id": record.get("set_id", ""),
        "effective_time": record.get("effective_time", ""),
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def build_documents() -> List[Document]:
    """
    Fetches all configured medicines and converts them into LangChain Documents.
    """

    documents: List[Document] = []

    for medicine in MEDICINES:
        display_name = medicine["display_name"]
        query_name = medicine["openfda_query_name"]

        print(f"[INFO] Fetching records for: {display_name}")

        records = fetch_medicine_records(query_name, limit=3)

        if not records:
            continue

        for record in records:
            document = create_document_from_record(record, medicine)
            documents.append(document)

        print(f"[INFO] Documents created so far: {len(documents)}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits large medicine documents into smaller chunks.

    Important:
    split_documents() preserves metadata from the original Document.
    So every chunk still knows its medicine_name, source, set_id, etc.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk-level metadata manually.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def main() -> None:
    print("[START] Medicine ingestion started")

    documents = build_documents()

    print(f"[INFO] Total full documents created: {len(documents)}")

    if not documents:
        raise ValueError("No medicine documents were created.")

    chunks = chunk_documents(documents)

    print(f"[INFO] Total chunks created: {len(chunks)}")

    save_vector_store(chunks)

    print("[DONE] Medicine ingestion completed")


if __name__ == "__main__":
    main()
