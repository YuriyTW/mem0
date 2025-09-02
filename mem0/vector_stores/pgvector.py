import json
import logging
from typing import List, Optional

from pydantic import BaseModel

# Try to import PostgreSQL clients in order: psycopg (psycopg3), psycopg2, asyncpg
# This allows flexibility in choosing the best client for specific use cases
try:
    import psycopg
    from psycopg import execute_values
    from psycopg.types.json import Json
    PSYCOPG_VERSION = 3
    HAS_PSYCOPG3 = True
except ImportError:
    HAS_PSYCOPG3 = False
    psycopg = None
    execute_values = None
    Json = None

try:
    import psycopg2
    from psycopg2.extras import execute_values as psycopg2_execute_values, Json as psycopg2_Json
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None
    psycopg2_execute_values = None
    psycopg2_Json = None

try:
    import asyncpg
    import asyncio
    import threading
    import json as json_module
    from concurrent.futures import ThreadPoolExecutor
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None
    asyncio = None
    threading = None
    ThreadPoolExecutor = None
    json_module = None

# Determine which client to use as default (priority: asyncpg > psycopg3 > psycopg2)
if HAS_ASYNCPG:
    PSYCOPG_VERSION = None
    DEFAULT_CLIENT = 'asyncpg'
    logger = logging.getLogger(__name__)
    logger.info("Using asyncpg for PostgreSQL connections")
elif HAS_PSYCOPG3:
    PSYCOPG_VERSION = 3
    DEFAULT_CLIENT = 'psycopg3'
    logger = logging.getLogger(__name__)
    logger.info("Using psycopg (psycopg3) for PostgreSQL connections")
elif HAS_PSYCOPG2:
    PSYCOPG_VERSION = 2
    DEFAULT_CLIENT = 'psycopg2'
    logger = logging.getLogger(__name__)
    logger.info("Using psycopg2 for PostgreSQL connections")
else:
    raise ImportError(
        "None of the supported PostgreSQL libraries are available. "
        "Please install one of: 'pip install asyncpg', 'pip install psycopg', or 'pip install psycopg2'."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


def format_vector_for_pg(vector):
    """Format a vector list for PostgreSQL vector type."""
    if isinstance(vector, list):
        # Convert list to PostgreSQL vector format: [1.0,2.0,3.0]
        return "[" + ",".join(map(str, vector)) + "]"
    return str(vector)

class AsyncPGWrapper:
    """
    Wrapper class to make asyncpg work in synchronous context.
    This class handles the event loop management and threading to ensure
    asyncpg operations can be used in synchronous code.
    """
    
    def __init__(self, connection_params):
        self.connection_params = connection_params
        self._loop = None
        self._thread = None
        self._connection = None
        self._pool = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._start_async_loop()
        
    def _start_async_loop(self):
        """Start the async event loop in a separate thread."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
            
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Wait for loop to be ready
        while self._loop is None:
            threading.Event().wait(0.01)
            
    def _run_async(self, coro):
        """Run an async coroutine in the async thread and return the result."""
        if self._loop is None:
            raise RuntimeError("Async loop not initialized")
            
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
        
    async def _connect_async(self):
        """Establish async connection."""
        if self._connection is None:
            # Disable prepared statement caching for compatibility with pgbouncer
            conn_params = self.connection_params.copy()
            conn_params['statement_cache_size'] = 0
            self._connection = await asyncpg.connect(**conn_params)
        return self._connection
        
    async def _execute_async(self, query, *args):
        """Execute a query asynchronously."""
        conn = await self._connect_async()
        return await conn.execute(query, *args)
        
    async def _fetch_async(self, query, *args):
        """Fetch query results asynchronously."""
        conn = await self._connect_async()
        return await conn.fetch(query, *args)
        
    async def _fetchone_async(self, query, *args):
        """Fetch one result asynchronously."""
        conn = await self._connect_async()
        return await conn.fetchrow(query, *args)
        
    def execute(self, query, *args):
        """Execute a query synchronously."""
        return self._run_async(self._execute_async(query, *args))
        
    def fetchall(self, query, *args):
        """Fetch all results synchronously."""
        return self._run_async(self._fetch_async(query, *args))
        
    def fetchone(self, query, *args):
        """Fetch one result synchronously."""
        return self._run_async(self._fetchone_async(query, *args))
        
    async def _executemany_async(self, query, param_list):
        """Execute many queries asynchronously."""
        conn = await self._connect_async()
        async with conn.transaction():
            for params in param_list:
                await conn.execute(query, *params)
                
    def executemany(self, query, param_list):
        """Execute many queries synchronously."""
        return self._run_async(self._executemany_async(query, param_list))
        
    def commit(self):
        """Asyncpg uses autocommit by default, so this is a no-op."""
        pass
        
    def close(self):
        """Close the connection and cleanup resources."""
        async def _close():
            if self._connection:
                await self._connection.close()
                self._connection = None
                
        if self._loop and self._connection:
            self._run_async(_close())
            
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=1)
        if self._executor:
            self._executor.shutdown(wait=True)
            
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class PGVector(VectorStoreBase):
    def __init__(
        self,
        dbname,
        collection_name,
        embedding_model_dims,
        user,
        password,
        host,
        port,
        diskann,
        hnsw,
        sslmode=None,
        connection_string=None,
        connection_pool=None,
        client_type=None,
    ):
        """
        Initialize the PGVector database.

        Args:
            dbname (str): Database name
            collection_name (str): Collection name
            embedding_model_dims (int): Dimension of the embedding vector
            user (str): Database user
            password (str): Database password
            host (str, optional): Database host
            port (int, optional): Database port
            diskann (bool, optional): Use DiskANN for faster search
            hnsw (bool, optional): Use HNSW for faster search
            sslmode (str, optional): SSL mode for PostgreSQL connection (e.g., 'require', 'prefer', 'disable')
            connection_string (str, optional): PostgreSQL connection string (overrides individual connection parameters)
            connection_pool (Any, optional): psycopg2 connection pool object (overrides connection string and individual parameters)
            client_type (str, optional): Specify which client to use: 'psycopg3', 'psycopg2', or 'asyncpg'. If None, uses default available client.
        """
        self.collection_name = collection_name
        self.use_diskann = diskann
        self.use_hnsw = hnsw
        self.embedding_model_dims = embedding_model_dims
        
        # Determine which client to use
        self.client_type = self._determine_client_type(client_type)
        
        # Initialize connection based on client type
        self._initialize_connection(
            dbname, user, password, host, port, sslmode,
            connection_string, connection_pool
        )
        
        collections = self.list_cols()
        if collection_name not in collections:
            self.create_col(embedding_model_dims)

    def _determine_client_type(self, client_type):
        """Determine which PostgreSQL client to use."""
        if client_type is not None:
            # Explicit client type requested
            if client_type == 'psycopg3' and not HAS_PSYCOPG3:
                raise ImportError("psycopg (psycopg3) not available. Install with 'pip install psycopg'")
            elif client_type == 'psycopg2' and not HAS_PSYCOPG2:
                raise ImportError("psycopg2 not available. Install with 'pip install psycopg2'")
            elif client_type == 'asyncpg' and not HAS_ASYNCPG:
                raise ImportError("asyncpg not available. Install with 'pip install asyncpg'")
            return client_type
        else:
            # Use default client
            return DEFAULT_CLIENT
            
    def _initialize_connection(self, dbname, user, password, host, port, sslmode, connection_string, connection_pool):
        """Initialize database connection based on client type."""
        # Connection setup with priority: connection_pool > connection_string > individual parameters
        if connection_pool is not None and self.client_type != 'asyncpg':
            # Use provided connection pool (not supported for asyncpg)
            if self.client_type == 'psycopg3':
                self.conn = connection_pool.getconn()
            else:  # psycopg2
                self.conn = connection_pool.getconn()
            self.connection_pool = connection_pool
            self.cur = self.conn.cursor()
            return
            
        # Prepare connection parameters
        if connection_string is not None:
            # Use connection string
            if sslmode:
                import re
                if 'sslmode=' in connection_string:
                    connection_string = re.sub(r'sslmode=[^ ]*', f'sslmode={sslmode}', connection_string)
                else:
                    connection_string = f"{connection_string} sslmode={sslmode}"
            
            if self.client_type == 'psycopg3':
                self.conn = psycopg.connect(connection_string)
                self.cur = self.conn.cursor()
            elif self.client_type == 'psycopg2':
                self.conn = psycopg2.connect(connection_string)
                self.cur = self.conn.cursor()
            else:  # asyncpg
                # Parse connection string for asyncpg
                conn_params = self._parse_connection_string(connection_string)
                self.conn = AsyncPGWrapper(conn_params)
                self.cur = None  # asyncpg doesn't use cursors
        else:
            # Use individual connection parameters
            conn_params = {
                'database' if self.client_type == 'asyncpg' else 'dbname': dbname,
                'user': user,
                'password': password,
                'host': host,
                'port': port
            }
            if sslmode:
                conn_params['sslmode'] = sslmode
            
            if self.client_type == 'psycopg3':
                # Fix parameter name for psycopg3
                conn_params['dbname'] = conn_params.pop('database', dbname)
                self.conn = psycopg.connect(**conn_params)
                self.cur = self.conn.cursor()
            elif self.client_type == 'psycopg2':
                # Fix parameter name for psycopg2  
                conn_params['dbname'] = conn_params.pop('database', dbname)
                self.conn = psycopg2.connect(**conn_params)
                self.cur = self.conn.cursor()
            else:  # asyncpg
                self.conn = AsyncPGWrapper(conn_params)
                self.cur = None  # asyncpg doesn't use cursors
                
        self.connection_pool = None
        
    def _parse_connection_string(self, connection_string):
        """Parse PostgreSQL connection string for asyncpg."""
        import re
        params = {}
        
        # Extract parameters from connection string
        patterns = {
            'host': r'host=([^\s]+)',
            'port': r'port=([^\s]+)',
            'database': r'(?:dbname|database)=([^\s]+)',
            'user': r'user=([^\s]+)', 
            'password': r'password=([^\s]+)',
            'sslmode': r'sslmode=([^\s]+)'
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, connection_string)
            if match:
                value = match.group(1)
                # Convert port to int if it's port
                if param == 'port':
                    value = int(value)
                params[param] = value
                
        return params

    def create_col(self, embedding_model_dims):
        """
        Create a new collection (table in PostgreSQL).
        Will also initialize vector search index if specified.

        Args:
            embedding_model_dims (int): Dimension of the embedding vector.
        """
        if self.client_type == 'asyncpg':
            # For asyncpg, execute directly on connection
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id UUID PRIMARY KEY,
                    vector vector({embedding_model_dims}),
                    payload JSONB
                );
            """
            )

            if self.use_diskann and embedding_model_dims < 2000:
                # Check if vectorscale extension is installed
                result = self.conn.fetchone("SELECT * FROM pg_extension WHERE extname = 'vectorscale'")
                if result:
                    # Create DiskANN index if extension is installed for faster search
                    self.conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.collection_name}_diskann_idx
                        ON {self.collection_name}
                        USING diskann (vector);
                    """
                    )
            elif self.use_hnsw:
                self.conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                    ON {self.collection_name}
                    USING hnsw (vector vector_cosine_ops)
                """
                )
        else:
            # For psycopg3 and psycopg2
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id UUID PRIMARY KEY,
                    vector vector({embedding_model_dims}),
                    payload JSONB
                );
            """
            )

            if self.use_diskann and embedding_model_dims < 2000:
                # Check if vectorscale extension is installed
                self.cur.execute("SELECT * FROM pg_extension WHERE extname = 'vectorscale'")
                if self.cur.fetchone():
                    # Create DiskANN index if extension is installed for faster search
                    self.cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.collection_name}_diskann_idx
                        ON {self.collection_name}
                        USING diskann (vector);
                    """
                    )
            elif self.use_hnsw:
                self.cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                    ON {self.collection_name}
                    USING hnsw (vector vector_cosine_ops)
                """
                )

        self.conn.commit()

    def insert(self, vectors, payloads=None, ids=None):
        """
        Insert vectors into a collection.

        Args:
            vectors (List[List[float]]): List of vectors to insert.
            payloads (List[Dict], optional): List of payloads corresponding to vectors.
            ids (List[str], optional): List of IDs corresponding to vectors.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        
        if self.client_type == 'asyncpg':
            # For asyncpg, use executemany with individual inserts
            insert_query = f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES ($1, $2::vector, $3::jsonb)"
            data = [(id, format_vector_for_pg(vector), json_module.dumps(payload)) for id, vector, payload in zip(ids, vectors, payloads)]
            self.conn.executemany(insert_query, data)
        else:
            # For psycopg3 and psycopg2
            json_payloads = [json.dumps(payload) for payload in payloads]
            data = [(id, vector, payload) for id, vector, payload in zip(ids, vectors, json_payloads)]
            
            if self.client_type == 'psycopg3':
                execute_values(
                    self.cur,
                    f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES %s",
                    data,
                )
            else:  # psycopg2
                psycopg2_execute_values(
                    self.cur,
                    f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES %s",
                    data,
                )
        
        self.conn.commit()

    def search(self, query, vectors, limit=5, filters=None):
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            vectors (List[float]): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (Dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                if self.client_type == 'asyncpg':
                    filter_conditions.append(f"payload->>'{k}' = ${len(filter_params) + 2}")  # +2 because $1 is vectors
                else:
                    filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        if self.client_type == 'asyncpg':
            # For asyncpg, use $1, $2, etc. placeholders
            base_params = [format_vector_for_pg(vectors)]
            param_counter = 2
            
            # Adjust filter clause for asyncpg syntax
            if filter_conditions:
                filter_conditions_asyncpg = []
                for k, v in filters.items():
                    filter_conditions_asyncpg.append(f"payload->>'{k}' = ${param_counter}")
                    base_params.append(str(v))
                    param_counter += 1
                filter_clause = "WHERE " + " AND ".join(filter_conditions_asyncpg)
            
            base_params.append(limit)
            
            query = f"""
                SELECT id, vector <=> $1::vector AS distance, payload
                FROM {self.collection_name}
                {filter_clause}
                ORDER BY distance
                LIMIT ${param_counter}
            """
            
            results = self.conn.fetchall(query, *base_params)
        else:
            # For psycopg3 and psycopg2
            self.cur.execute(
                f"""
                SELECT id, vector <=> %s::vector AS distance, payload
                FROM {self.collection_name}
                {filter_clause}
                ORDER BY distance
                LIMIT %s
            """,
                (vectors, *filter_params, limit),
            )
            results = self.cur.fetchall()

        # Parse JSON payloads for asyncpg (returns as string)
        if self.client_type == 'asyncpg':
            return [OutputData(id=str(r[0]), score=float(r[1]), 
                              payload=json_module.loads(r[2]) if isinstance(r[2], str) else r[2]) 
                    for r in results]
        else:
            return [OutputData(id=str(r[0]), score=float(r[1]), payload=r[2]) for r in results]

    def delete(self, vector_id):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        if self.client_type == 'asyncpg':
            self.conn.execute(f"DELETE FROM {self.collection_name} WHERE id = $1", vector_id)
        else:
            self.cur.execute(f"DELETE FROM {self.collection_name} WHERE id = %s", (vector_id,))
        self.conn.commit()

    def update(self, vector_id, vector=None, payload=None):
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update.
            vector (List[float], optional): Updated vector.
            payload (Dict, optional): Updated payload.
        """
        if vector:
            if self.client_type == 'asyncpg':
                self.conn.execute(
                    f"UPDATE {self.collection_name} SET vector = $1::vector WHERE id = $2",
                    format_vector_for_pg(vector), vector_id
                )
            else:
                self.cur.execute(
                    f"UPDATE {self.collection_name} SET vector = %s WHERE id = %s",
                    (vector, vector_id),
                )
        if payload:
            if self.client_type == 'asyncpg':
                self.conn.execute(
                    f"UPDATE {self.collection_name} SET payload = $1::jsonb WHERE id = $2",
                    json_module.dumps(payload), vector_id
                )
            elif self.client_type == 'psycopg3':
                self.cur.execute(
                    f"UPDATE {self.collection_name} SET payload = %s WHERE id = %s",
                    (Json(payload), vector_id),
                )
            else:  # psycopg2
                self.cur.execute(
                    f"UPDATE {self.collection_name} SET payload = %s WHERE id = %s",
                    (psycopg2_Json(payload), vector_id),
                )
        self.conn.commit()

    def get(self, vector_id) -> OutputData:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            OutputData: Retrieved vector.
        """
        if self.client_type == 'asyncpg':
            result = self.conn.fetchone(
                f"SELECT id, vector, payload FROM {self.collection_name} WHERE id = $1",
                vector_id
            )
        else:
            self.cur.execute(
                f"SELECT id, vector, payload FROM {self.collection_name} WHERE id = %s",
                (vector_id,),
            )
            result = self.cur.fetchone()
        
        if not result:
            return None
        
        # Parse JSON payload for asyncpg
        payload = result[2]
        if self.client_type == 'asyncpg' and isinstance(payload, str):
            payload = json_module.loads(payload)
            
        return OutputData(id=str(result[0]), score=None, payload=payload)

    def list_cols(self) -> List[str]:
        """
        List all collections.

        Returns:
            List[str]: List of collection names.
        """
        if self.client_type == 'asyncpg':
            results = self.conn.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        else:
            self.cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            results = self.cur.fetchall()
        return [row[0] for row in results]

    def delete_col(self):
        """Delete a collection."""
        if self.client_type == 'asyncpg':
            self.conn.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
        else:
            self.cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
        self.conn.commit()

    def col_info(self):
        """
        Get information about a collection.

        Returns:
            Dict[str, Any]: Collection information.
        """
        if self.client_type == 'asyncpg':
            result = self.conn.fetchone(
                f"""
                SELECT 
                    table_name, 
                    (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                    (SELECT pg_size_pretty(pg_total_relation_size('{self.collection_name}'))) as total_size
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = $1
            """,
                self.collection_name
            )
        else:
            self.cur.execute(
                f"""
                SELECT 
                    table_name, 
                    (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                    (SELECT pg_size_pretty(pg_total_relation_size('{self.collection_name}'))) as total_size
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = %s
            """,
                (self.collection_name,),
            )
            result = self.cur.fetchone()
        return {"name": result[0], "count": result[1], "size": result[2]}

    def list(self, filters=None, limit=100):
        """
        List all vectors in a collection.

        Args:
            filters (Dict, optional): Filters to apply to the list.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            List[OutputData]: List of vectors.
        """
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                if self.client_type == 'asyncpg':
                    filter_conditions.append(f"payload->>'{k}' = ${len(filter_params) + 1}")
                    filter_params.append(str(v))
                else:
                    filter_conditions.append("payload->>%s = %s")
                    filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        if self.client_type == 'asyncpg':
            query = f"""
                SELECT id, vector, payload
                FROM {self.collection_name}
                {filter_clause}
                LIMIT ${len(filter_params) + 1}
            """
            filter_params.append(limit)
            results = self.conn.fetchall(query, *filter_params)
        else:
            query = f"""
                SELECT id, vector, payload
                FROM {self.collection_name}
                {filter_clause}
                LIMIT %s
            """
            self.cur.execute(query, (*filter_params, limit))
            results = self.cur.fetchall()
        
        # Parse JSON payloads for asyncpg
        if self.client_type == 'asyncpg':
            parsed_results = []
            for r in results:
                payload = r[2]
                if isinstance(payload, str):
                    payload = json_module.loads(payload)
                parsed_results.append(OutputData(id=str(r[0]), score=None, payload=payload))
            return [parsed_results]
        else:
            return [[OutputData(id=str(r[0]), score=None, payload=r[2]) for r in results]]

    def __del__(self):
        """
        Close the database connection when the object is deleted.
        """
        if hasattr(self, "cur") and self.cur:
            self.cur.close()
        if hasattr(self, "conn"):
            if hasattr(self, "connection_pool") and self.connection_pool is not None:
                # Return connection to pool instead of closing it
                self.connection_pool.putconn(self.conn)
            elif self.client_type == 'asyncpg':
                # AsyncPGWrapper handles its own cleanup
                self.conn.close()
            else:
                # Close the connection directly for psycopg3/psycopg2
                self.conn.close()

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.embedding_model_dims)