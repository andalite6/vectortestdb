import pandas as pd
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
import os

# Configure the client
client = OpenSearch(
    hosts = [{'host': 'search-ethics-db-o2cpmhtgzof4pfkluqlhqk6uwu.us-east-1.es.amazonaws.com', 'port': 443}],
    http_auth = ('impactuser', 'Impactadmin1@'),
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

# Define index name and body BEFORE checking if it exists
index_name = 'vector-index'
index_body = {
    'settings': {
        'index.knn': True  # Enable k-NN for this index
    },
    'mappings': {
        'properties': {
            'vector_field': {
                'type': 'knn_vector',
                'dimension': 384,
                'method': {
                    'name': 'hnsw',
                    'space_type': 'l2',  # Using l2 distance (Euclidean)
                    'engine': 'faiss'
                }
            },
            'metadata_field': {
                'type': 'text'
            }
        }
    }
}

# Read the CSV file
csv_path = os.path.expanduser("/Users/macuser1/Downloads/ethics dashboard/vectordb/vector_data.csv")
df = pd.read_csv(csv_path)

# Check if index exists and delete it if it does
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)
    print(f"Deleted existing index: {index_name}")

# Create the index
response = client.indices.create(index=index_name, body=index_body)
print("Index creation response:", response)

# Initialize vector_data for the final query
final_vector_data = None

# Process each row in the CSV
for index, row in df.iterrows():
    # If your CSV already contains vector data in columns
    if all(f'vector_{i}' in df.columns for i in range(1, 385)):
        # Extract vector values from columns
        vector_data = [float(row[f'vector_{i}']) for i in range(1, 385)]
    else:
        # If CSV doesn't have pre-computed vectors, generate from text
        # Combine all fields to create text for embedding
        text_data = ' '.join(row.astype(str).values)
        # In production, use a real embedding model here
        vector_data = np.random.rand(384).tolist()  # Replace with real embeddings
    
    # Save the last vector for the final query
    final_vector_data = vector_data
    
    # Create document including vector and metadata
    document = {
        'vector_field': vector_data,
        'metadata_field': f'Row {index} from vector_data.csv'
    }
    
    # Add all columns from CSV as fields
    for column in df.columns:
        # Skip vector columns if they exist
        if not column.startswith('vector_'):
            document[column] = str(row[column])
    
    # Index the document
    response = client.index(
        index = index_name,
        body = document,
        refresh = True
    )
    
    # Print progress periodically
    if index % 10 == 0:
        print(f"Indexed document {index} with ID: {response['_id']}")

print("CSV data ingestion complete!")

# Make sure we have a vector to query with
if final_vector_data is None:
    final_vector_data = np.random.rand(384).tolist()

# Perform k-NN search using the last processed vector
query = {
    'size': 10,
    'query': {
        'knn': {
            'vector_field': {
                'vector': final_vector_data,
                'k': 5
            }
        }
    }
}

response = client.search(
    body = query,
    index = index_name
)

print(response['hits']['hits'])