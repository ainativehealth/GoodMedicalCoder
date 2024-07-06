from ragatouille import RAGPretrainedModel
import csv

# Load the RAG model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Read ICD-10 codes and descriptions from the CSV file
icd10_documents = []
document_ids = []
document_metadatas = []

with open('ICD-10.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t')
    for i, row in enumerate(csv_reader):
        if len(row) < 2:
            print(f"Skipping row {i} due to incorrect format: {row}")
            continue
        code = row[0]
        description = ' '.join(row[1:])  # Join all remaining fields for the description
        icd10_documents.append(description)
        document_ids.append(code)
        document_metadatas.append({"type": "ICD-10", "code": code})

print(f"Processed {len(icd10_documents)} valid rows")

# Create the index
index_path = RAG.index(
    index_name="icd10_index",
    collection=icd10_documents,
    document_ids=document_ids,
    document_metadatas=document_metadatas,
)

print(f"Index created and saved at: {index_path}")

