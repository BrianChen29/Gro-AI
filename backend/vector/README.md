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

For callers that already have multiple product queries,
`search_similar_items_batch` embeds them in one provider request and performs
the independent vector-store searches with bounded concurrency within that
batch (five at a time by default). Its outer result order matches the input
query order, each inner result keeps the store ranking, and the same optional
score threshold is applied to every query. The existing single-query API
remains available.

## Initial Pinecone Sync

The initial sync reuses the existing SQLite vectors. It does not call the
OpenAI embeddings API.

First configure `backend/.env`:

```text
VECTOR_STORE_BACKEND=pinecone
VECTOR_SEARCH_MIN_SCORE=
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

## Retrieval Evaluation

`retrieval_eval` measures the current `VectorStore` backend against
human-labeled query-to-product relevance judgments. This is a local information
retrieval evaluator, not an OpenAI Evals job: OpenAI is used only to embed the
queries, while Gro AI calculates the retrieval metrics itself.

Create a JSON file containing real item IDs from the MySQL catalog:

```json
{
  "schema_version": 1,
  "description": "Human-reviewed grocery retrieval cases",
  "cases": [
    {
      "case_id": "oat-milk-intent",
      "query": "unsweetened oat milk",
      "relevant_item_ids": [123]
    },
    {
      "case_id": "birthday-cake-intent",
      "query": "chocolate birthday cake",
      "relevant_item_ids": [456, 789]
    }
  ]
}
```

The IDs above are placeholders, not Gro AI catalog IDs. Each case must contain
at least one human-reviewed relevant ID; include every product that would be a
valid result, rather than labeling only the first convenient match.

From `backend`, validate the file without an embedding cache, API key, or
vector store connection:

```bash
python -m vector.retrieval_eval \
  --cases vector/evaluation_cases.json \
  --validate-only
```

Run the evaluation against the backend selected by `VECTOR_STORE_BACKEND`:

```bash
python -m vector.retrieval_eval \
  --cases vector/evaluation_cases.json \
  --top-k 5
```

The command batches all case queries into one OpenAI embeddings request, then
searches the selected store once per case. It reports:

- **Hit Rate@K:** fraction of cases with at least one relevant result;
- **Mean Precision@K:** average relevant results divided by `K`;
- **Mean Recall@K:** average fraction of all labeled relevant IDs retrieved;
- **MRR@K:** average reciprocal rank of the first relevant result.

Every returned candidate retains its rank, vector similarity score, relevance
label, and metadata. When `--min-score` is omitted, the report also tests each
observed cosine score as an inclusive global cutoff and recommends the one with
the highest F1. Ties prefer higher recall, then higher precision, then the lower
cutoff. A recommendation is intentionally unavailable unless the retrieved
candidate set contains both relevant and non-relevant examples.

The threshold analysis uses micro-averaged binary counts across all cases:
retained relevant hits are true positives, retained non-relevant hits are false
positives, and labeled relevant IDs that are filtered out or absent from the
top-K candidates are false negatives. This precision is different from
Precision@K: Precision@K always divides by K, while threshold precision divides
by the number of candidates retained by the cutoff.

### Calibrate, validate, then enable a threshold

Split the reviewed cases into a calibration file and a separate held-out file
before choosing a threshold. Use the same `--top-k` value as production. First
generate a recommendation from only the calibration cases:

```bash
python -m vector.retrieval_eval \
  --cases vector/evaluation_calibration.json \
  --top-k 10 > calibration-report.json
```

Copy `score_threshold_analysis.recommended_min_score` from that report and
evaluate that fixed value on the held-out cases:

```bash
python -m vector.retrieval_eval \
  --cases vector/evaluation_held_out.json \
  --top-k 10 \
  --min-score 0.72 > held-out-report.json
```

The number above is only an example; do not use it as Gro AI's cutoff. If the
held-out tradeoff is acceptable, set the reviewed value in `backend/.env`:

```text
VECTOR_SEARCH_MIN_SCORE=0.72
```

Leaving the setting blank preserves the existing top-K behavior. A configured
cutoff is applied after vector-store retrieval and may return fewer than K
items. Recalibrate after changing the embedding model, vector backend/index,
catalog, relevance labels, or production top-K. The evaluator never writes the
recommendation into `.env`, Pinecone, or application state.

This is a global relevance gate, not a reranker: it removes weak matches but
does not change the order of retained results. A future reranker would need its
own labeled features and evaluation rather than using aggregate metrics as
query-time inputs.

The regular evaluation is read-only but does call OpenAI and the selected
vector store. Memory-mode evaluation fails loudly if its SQLite cache is
missing or empty. Optional case `filters` are passed to `VectorStore.search`;
omit them when comparing against a legacy SQLite cache that has no metadata.
The case count is limited to 2,048 because all queries are embedded in one
batch. Standard output is JSON, so a report can be saved with shell
redirection for later comparison.

Keep the reviewed case file under version control alongside the catalog version
it describes; otherwise an ID change can silently make relevance labels stale.
