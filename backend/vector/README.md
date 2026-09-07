# Catalog Vector Retrieval

Gro AI keeps product records in MySQL and retrieves their IDs through a
replaceable `VectorStore` backend:

- `memory`: loads the SQLite embedding cache and performs exact cosine search.
- `pinecone`: queries a pre-existing Pinecone vector index asynchronously.

Pinecone stores only the catalog ID, vector, and stable retrieval metadata
(`title` and `sub_category`). MySQL remains the source of truth for price,
rating, and other product fields.

Indexing and query-time retrieval must use the same embedding model. The
default is `text-embedding-3-large`; configure it once with:

```text
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

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
vector ID, so rerunning replaces records with the same IDs.

## Incremental Catalog Updates

After the initial sync, use `catalog_update` when a product's `title` or
`sub_category` changes, when a new product is added, or after a product is
deleted. These commands are also dry-run by default.

### Add or update products

First commit the product changes to MySQL. Then inspect the intended targets:

```bash
python -m vector.catalog_update upsert 42 57
```

The dry run verifies that every requested ID exists, reports the configured
embedding model and cache dimension, and makes no OpenAI or Pinecone request.
Execute only after reviewing that report:

```bash
python -m vector.catalog_update upsert 42 57 --execute
```

Execution builds one canonical string per item (`title | sub_category`), sends
the strings to the OpenAI embeddings API as one batch, upserts the resulting
vectors and metadata to Pinecone, and replaces the matching rows in the local
SQLite cache. A batch is limited to 2,048 item IDs.

### Delete products

Delete the product rows from MySQL first, then validate the vector deletion:

```bash
python -m vector.catalog_update delete 42 57
```

The command refuses to execute while any requested ID still exists in MySQL.
After the dry run reports `"ready_to_execute": true`, apply it:

```bash
python -m vector.catalog_update delete 42 57 --execute
```

Execution deletes the IDs from Pinecone and the local SQLite cache. Both
upserts and deletes are idempotent, so the same command can be rerun if a local
cache update fails after Pinecone succeeds.

`--execute` requires `VECTOR_STORE_BACKEND=pinecone`. Upsert execution also
requires `OPENAI_API_KEY`; delete execution does not call OpenAI. Use
`--embeddings-db /absolute/path/to/embeddings.sqlite` to override the cache for
either operation. If the production fallback cache is stored in GCS, upload
the updated SQLite file after a successful incremental command.
