# Catalog Vector Retrieval

Gro AI keeps product records in MySQL and retrieves their IDs through a
replaceable `VectorStore` backend:

- `memory`: loads the SQLite embedding cache and performs exact cosine search.
- `pinecone`: queries a pre-existing Pinecone vector index asynchronously.

Pinecone stores only the catalog ID, vector, and stable retrieval metadata
(`title` and `sub_category`). MySQL remains the source of truth for price,
rating, and other product fields.

## Initial Pinecone Sync

The initial sync reuses the existing SQLite vectors. It does not call the
OpenAI embeddings API.

First configure `backend/.env`:

```text
VECTOR_STORE_BACKEND=pinecone
EMBEDDINGS_DB_PATH=vector/embeddings.sqlite
PINECONE_API_KEY=...
PINECONE_INDEX_HOST=...
PINECONE_NAMESPACE=catalog
PINECONE_BATCH_SIZE=50
```

From the `backend` directory, run the read-only validation first:

```bash
python -m vector.catalog_sync
```

Dry run is the default. Its JSON output reports:

- the number of MySQL catalog items and cached embeddings;
- IDs missing an embedding;
- cached IDs no longer present in MySQL;
- the vector dimension required by the Pinecone index.

Create the Pinecone index before execution. It must use cosine similarity and
the `vector_dimension` reported by dry run. Then execute the upsert:

```bash
python -m vector.catalog_sync --execute
```

Execution stops before writing if any catalog item lacks a cached vector. Use
`--allow-partial` only when intentionally creating an incomplete index:

```bash
python -m vector.catalog_sync --execute --allow-partial
```

You can override the cache path for one run:

```bash
python -m vector.catalog_sync \
  --embeddings-db /absolute/path/to/embeddings.sqlite
```

The operation is repeatable: Pinecone upserts use the grocery item ID as the
vector ID, so rerunning replaces records with the same IDs. This version only
implements initial full-catalog synchronization; targeted re-embedding and
deletion synchronization are separate follow-up work.
