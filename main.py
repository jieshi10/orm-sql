from typing import List, Dict, Tuple, Optional, Any

from sentence_transformers import SentenceTransformer, util
import glob
import json
from tqdm import tqdm

import openai

from threading import Lock, BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor

from schema_free_utils_2 import get_tables_and_columns
from sql_utils_2 import build_sql, dump_db_json_schema, set_identifier
from sqlglot import parse_one, exp

import nltk
import re

import sqlite3

import time

import os

from copy import deepcopy


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ZERO_SHOT_CHAT = 'zero_shot_chat'
FEW_SHOT_COMPLETION = 'few_shot_completion'

SQL_MODE = 'SQL_MODE'

DB_SQLITE = 'DB_SQLITE'
DB_PG = 'DB_PG'
DB_MSSQL = 'DB_MSSQL'
DB_ORACLE = 'DB_ORACLE'
DB_MYSQL = 'DB_MYSQL'


###########################################################
# CONFIG

# USE_EXT = True
USE_EXT = False

# MAX_VALUES = 0
MAX_VALUES = 1

# FEW_SHOT_K = 16
FEW_SHOT_K = 8
# FEW_SHOT_K = 4
# FEW_SHOT_K = 3
# FEW_SHOT_K = 2
# FEW_SHOT_K = 1
# FEW_SHOT_K = 0

MODE = ZERO_SHOT_CHAT
# MODE = FEW_SHOT_COMPLETION

METHOD = 'sqlalchemy'

GEN_MODE: str=SQL_MODE

# USE_QUESTION_SKELETON = False
USE_QUESTION_SKELETON = True

USE_SQL_SKELETON = False

# MAX_ROUNDS = 1
MAX_ROUNDS = 10  # max rounds for generation

DB_NAME: str='bird'
# DB_NAME: str='spider'

# DEBUG = True
DEBUG = False

BACKEND_DB = DB_SQLITE
# BACKEND_DB = DB_PG
# BACKEND_DB = DB_MSSQL
# BACKEND_DB = DB_ORACLE
# BACKEND_DB = DB_MYSQL

MAN_FILE_SUFFIX: str='dev'

if DEBUG:
    if MAN_FILE_SUFFIX is None:
        MAN_FILE_SUFFIX = 'debug'
    else:
        MAN_FILE_SUFFIX += '-debug'


if BACKEND_DB == DB_PG:
    import psycopg2

    db_conn_config = dict(
        host="127.0.0.1", port=5432, user="postgres", password="YOUR_PASSWORD"
    )
elif BACKEND_DB == DB_MSSQL:
    import pyodbc

    db_conn_config = dict(
        dsn="MSSQLServerDatabase", user="sa", password="YOUR_PASSWORD"
    )
elif BACKEND_DB == DB_ORACLE:
    import oracledb

    db_conn_config = dict(
        host="127.0.0.1", port=1521, user="test_user", password="YOUR_PASSWORD"
    )
elif BACKEND_DB == DB_MYSQL:
    import pymysql

    db_conn_config = dict(
        host="127.0.0.1", port=3306, user="root", password="YOUR_PASSWORD"
    )


if DB_NAME == 'bird':
    # bird
    DB_SET_PATH: str='benchmark/BIRD/dev/dev_databases/*'
    TRAINING_SET_PATHS: List[str]=[
        'data/bird-train-orm-sqlite.json',
        # 'data/bird-train-orm-pg.json',
        # 'data/bird-train-orm-mssql.json',
        # 'data/bird-train-orm-oracle.json',
        # 'data/bird-train-orm-mysql.json',
    ]
    DB_DESCRIPTION_PATH: Optional[str]=None
    DEV_SET_PATH: str='benchmark/BIRD/dev/dev.json'
    if BACKEND_DB in [DB_SQLITE, DB_MYSQL]:
        set_identifier('`')
    else:
        set_identifier('"')
    WITH_GOLD_SQL = False
elif DB_NAME == 'spider':
    # spider
    DB_SET_PATH: str='benchmark/spider/database/*'
    TRAINING_SET_PATHS: List[str]=[
        'data/spider-train-others-orm-sqlite.json',
        'data/spider-train-spider-orm-sqlite.json',
    ]
    DB_DESCRIPTION_PATH: Optional[str]=None
    DEV_SET_PATH: str='benchmark/spider/dev.json'
    if BACKEND_DB in [DB_SQLITE, DB_MYSQL]:
        set_identifier('')
    else:
        set_identifier('"')
    WITH_GOLD_SQL = False
else:
    raise ValueError(f'Unrecognized DB_NAME: {DB_NAME}')

###########################################################


def get_file_suffix():
    if MODE == ZERO_SHOT_CHAT:
        suf = 'zs'
        if FEW_SHOT_K > 0:
            suf += f'{FEW_SHOT_K}'
    elif MODE == FEW_SHOT_COMPLETION:
        suf = f'fs{FEW_SHOT_K}'
    else:
        raise ValueError(f'invalid MODE: {MODE}')
    pre = f'{METHOD}'
    if USE_EXT:
        pre = f'ext-{pre}'
    if MAX_ROUNDS > 1:
        pre += f'-L{MAX_ROUNDS}'
    if USE_QUESTION_SKELETON:
        pre += '-q_skltn'
    if USE_SQL_SKELETON:
        pre += '-sql_skltn'
    suffix = f'{pre}-{suf}'
    if MAN_FILE_SUFFIX is not None and MAN_FILE_SUFFIX != '':
        suffix += f'-{MAN_FILE_SUFFIX}'
    if BACKEND_DB == DB_PG:
        suffix += '-pg'
    elif BACKEND_DB == DB_MSSQL:
        suffix += '-mssql'
    elif BACKEND_DB == DB_ORACLE:
        suffix += '-oracle'
    elif BACKEND_DB == DB_MYSQL:
        suffix += '-mysql'
    return suffix


def get_database_name(db: str):
    if USE_EXT:
        return f'{db}_ext'
    else:
        return db


###########################################################


def count_select(sql: str):
    return len(list(parse_one(sql).find_all(exp.Select)))


# t123
def possibly_temp_table(name: str):
    name = name.lower()
    return name.startswith('t') and name[1:].isnumeric()


def infer_schemas_based_on_sql(sql: str, schema: list=None) -> Tuple[List[str], Dict[str, List[str]]]:
    referenced_columns = get_tables_and_columns(sql, schemas=schema)
    referenced_columns = {
        tbl_name: columns
        for tbl_name, columns in referenced_columns.items()
        if not possibly_temp_table(tbl_name)
    }
    res = []
    for tbl_name, columns in referenced_columns.items():
        # res.append(build_sql(tbl_name, [(c, 'TEXT') for c in columns], [], []))
        res.append(tbl_name + '(' + ','.join([c for c in columns]) + ')')
    return res, referenced_columns


def plain_table_schema(table: dict) -> str:
    return 'Schema: ' + table['name'] + '(' + ', '.join([c for c, _ in table['columns']]) + ')'


def format_db_value(v: Any):
    if v is None:
        return 'NULL'
    else:
        return repr(v)


def format_db_values(values: Optional[List[Any]]):
    if values is None or len(values) == 0:
        return ''
    else:
        return f"Examples: {', '.join([format_db_value(v) for v in values])}"


def format_comment(comment: Optional[str], values: Optional[List[Any]], enable_comment=True, enable_values=True):
    if not enable_comment:
        comment = None
    if not enable_values:
        values = None
    if comment is None and (values is None or len(values) == 0):
        return None
    elif comment is None:
        return format_db_values(values)
    elif values is None:
        return comment
    else:
        return f"{comment} {format_db_values(values)}"


def plain_column(table_name: str, column_name: str, column_type: str, comment: Optional[str]=None, values: Optional[List[Any]]=None) -> str:
    text = f"Table: {table_name}, Column: {column_name}"
    if comment is not None:
        text += f"\n\nDescription:\n{comment}"
        if values is not None:
            text += "\n"
    if values is not None and len(values) > 0:
        text += "\n" + format_db_values(values)
    return text


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


def timeout_after(seconds: float):
    from multiprocessing import Process, Manager

    def func_wrapper(fn):
        
        def wrapper(*args, **kwargs):

            with Manager() as mgr:
                res = mgr.dict()
            
                def f():
                    try:
                        res['ret'] = fn(*args, **kwargs)
                    except Exception as e:
                        res['exc'] = e
                
                p = Process(target=f)
                p.start()
                p.join(seconds)
                if p.exitcode is None:
                    p.terminate()
                    raise TimeoutError('timeout')
                else:
                    if 'ret' in res:
                        return res['ret']
                    else:
                        raise res['exc']

        return wrapper

    return func_wrapper


@timeout_after(10)
def exec_on_db(sqlite_path: str, query: str) -> Tuple[bool, Any]:
    if BACKEND_DB == DB_SQLITE:
        cursor = get_cursor_from_path(sqlite_path)
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return True, result
        except Exception as e:
            return False, e
        finally:
            cursor.close()
            cursor.connection.close()
    elif BACKEND_DB == DB_PG:
        with psycopg2.connect(dbname=sqlite_path, **db_conn_config) as conn:
            conn.set_session(autocommit=False, readonly=True)
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    return True, result
                except Exception as e:
                    return False, e
    elif BACKEND_DB == DB_MSSQL:
        with pyodbc.connect(f"DSN={db_conn_config['dsn']};UID={db_conn_config['user']};PWD={db_conn_config['password']};DATABASE={sqlite_path}") as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    return True, result
                except Exception as e:
                    return False, e
    elif BACKEND_DB == DB_ORACLE:
        with oracledb.connect(user=db_conn_config['user'], password=db_conn_config['password'], host=db_conn_config['host'], port=db_conn_config['port'], service_name=sqlite_path) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    return True, result
                except Exception as e:
                    return False, e
    elif BACKEND_DB == DB_MYSQL:
        with pymysql.connect(user=db_conn_config['user'], password=db_conn_config['password'], host=db_conn_config['host'], port=db_conn_config['port'], database=sqlite_path) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    return True, result
                except Exception as e:
                    return False, e
    else:
        raise ValueError(f'invalid BACKEND_DB: {BACKEND_DB}')


def check_sql_executability(generated_sql: str, db: str):
    if generated_sql.strip() == "":
        return "Error: empty string"
    try:
        # use `EXPLAIN QUERY PLAN` to avoid actually executing
        if BACKEND_DB == DB_SQLITE:
            explain = "EXPLAIN QUERY PLAN "
        elif BACKEND_DB == DB_PG:
            explain = "EXPLAIN "
        elif BACKEND_DB == DB_MSSQL:  # you have to actually execute the query
            explain = ""
        elif BACKEND_DB == DB_ORACLE:
            explain = ""
        elif BACKEND_DB == DB_MYSQL:
            explain = "EXPLAIN "
        else:
            raise ValueError(f'invalid BACKEND_DB: {BACKEND_DB}')
        success, res = exec_on_db(db, explain + generated_sql)
        if success:
            execution_error = None
        else:
            execution_error = str(res)
        return execution_error
    except Exception as e:
        return str(e)


# extract the skeleton of the input question
def extract_question_skeleton(text):
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))

    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ["NN", "NNP", "NNS", "NNPS", "CD", "SYM", "FW", "IN"]:
            output_tokens.append("_")
        elif token in ["$", "''", "(", ")", ",", "--", ".", ":"]:
            pass
        else:
            output_tokens.append(token)

    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while "_ _" in text_skeleton:
        text_skeleton = text_skeleton.replace("_ _", "_")
    while "_ , _" in text_skeleton:
        text_skeleton = text_skeleton.replace("_ , _", "_")

    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]

    return text_skeleton


def get_query_question(question: str):
    if DB_NAME == 'bird':
        if ' Hint: ' in question:
            question = question[:question.rindex(' Hint: ')]
    if USE_QUESTION_SKELETON:
        return extract_question_skeleton(question)
    return question


def get_query_sql(sql: str):
    if USE_SQL_SKELETON:
        assert False, 'not ready'
    return sql


def get_query_code(code: str):
    if USE_SQL_SKELETON:
        assert False, 'not ready'
    return code


class LLMClient:
    
    def __init__(self,
                 host: str='localhost',
                 port: int=8080,
                 model: str=None,
                 base_url: str=None,
                 api_key: str="empty",
        ):
        if base_url is None:
            assert host is not None and port is not None
            base_url = f"http://{host}:{port}/v1"
        self.client = openai.Client(
            api_key=api_key,
            base_url=base_url,
        )
        if model is None:
            # List models API
            models = self.client.models.list()
            assert len(models.data) == 1
            # print("Models:", models)
            self.model = models.data[0].id
        else:
            self.model = model
        print("Use Model:", self.model)
    
    def chat(self, **kwargs):
        res = self.client.chat.completions.create(
            model=self.model, **kwargs)
        return [choice.message.content for choice in res.choices], dict(res.usage)

    def complete(self, **kwargs):
        res = self.client.completions.create(
            model=self.model, **kwargs)
        return [choice.text for choice in res.choices], dict(res.usage)


class SingleDBColumnRetriever:
    
    def __init__(self, global_retriever: 'Retriever', col_schema):
        self.global_retriever = global_retriever
        self.lock = global_retriever.lock
        self.embedder = global_retriever.embedder
        global_corpus_map = global_retriever.column_corpus_map

        self.corpus = [plain_column(*c) for c in col_schema]
        self.corpus_full = col_schema
        self.corpus_name = [f'"{t}"."{c}"' for t, c, _, _, _ in col_schema]

        self.corpus_global_idx = [global_corpus_map[c] for c in self.corpus]

    @property
    def corpus_embeddings(self):
        return self.global_retriever.column_corpus_embeddings[self.corpus_global_idx]

    def batch_search_schemas(self, s: List[str], top_k: int):
        with self.lock:
            query_embedding = self.embedder.encode(s, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
            assert len(hits) == len(s)
            schemas = [[(
                self.corpus_name[hit['corpus_id']],
                self.corpus[hit['corpus_id']],
                self.corpus_full[hit['corpus_id']],
                hit['score'],
                hit['corpus_id'],
            ) for hit in item] for item in hits]
            return schemas


class SingleDBTableRetriever:
    
    def __init__(self, global_retriever: 'Retriever', schema, descriptions):
        self.global_retriever = global_retriever
        self.lock = global_retriever.lock
        self.embedder = global_retriever.embedder
        global_corpus_map = global_retriever.table_corpus_map

        self.corpus = [plain_table_schema(t) for t in schema]
        # self.corpus_full = [t['sql'] for t in schema]
        self.corpus_full = [build_sql(t['name'], t['columns'], t['primary_keys'], t['foreign_keys'], descriptions.get(t['name'].lower(), None)) for t in schema]
        self.corpus_name = [t['name'] for t in schema]
        self.corpus_name_idx = {n.lower(): i for i, n in enumerate(self.corpus_name)}

        self.corpus_global_idx = [global_corpus_map[t] for t in self.corpus]

    @property
    def corpus_embeddings(self):
        return self.global_retriever.table_corpus_embeddings[self.corpus_global_idx]

    def batch_search_schemas(self, s: List[str], top_k: int):
        with self.lock:
            query_embedding = self.embedder.encode(s, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
            assert len(hits) == len(s)
            schemas = [[(
                self.corpus_name[hit['corpus_id']],
                self.corpus[hit['corpus_id']],
                self.corpus_full[hit['corpus_id']],
                hit['score']
            ) for hit in item] for item in hits]
            return schemas


class ExampleRetriever:
    
    def __init__(self, global_retriever: 'Retriever', training_set_paths: List[str]):
        self.global_retriever = global_retriever
        self.lock = global_retriever.lock
        self.embedder = global_retriever.embedder

        print('loading training examples')
        self.example = []
        self.example_corpus_by_question = []
        self.example_corpus_by_sql = []
        self.example_corpus_by_code = []
        for filename in training_set_paths:
            with open(filename, encoding='utf-8') as f:
                print('reading:', filename)
                data = json.load(f)
                for r in data:
                    nl = r['question']
                    if 'evidence' in r:
                        nl += ' Hint: ' + r['evidence']
                    if 'query' in r and 'SQL' in r:
                        sql = r['SQL']
                        code = r['query']
                    elif 'query' in r:
                        sql = r['query']
                        code = None
                    elif 'SQL' in r:
                        sql = r['SQL']
                        code = None
                    else:
                        raise ValueError('no sql')
                    nl_q = get_query_question(nl)
                    sql_q = get_query_sql(sql)
                    e = {'nl': nl, 'nl_q': nl_q, 'sql': sql, 'sql_q': sql_q}
                    if code is not None:
                        e['code'] = code
                        code_q = get_query_code(code)
                        e['code_q'] = code_q
                        self.example_corpus_by_code.append(code_q)
                    self.example.append(e)
                    self.example_corpus_by_question.append(nl_q)
                    self.example_corpus_by_sql.append(sql_q)
        self.example_corpus_by_question_embeddings = self.embedder.encode(self.example_corpus_by_question, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        self.example_corpus_by_sql_embeddings = self.embedder.encode(self.example_corpus_by_sql, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        if len(self.example_corpus_by_code) > 0:
            self.example_corpus_by_code_embeddings = self.embedder.encode(self.example_corpus_by_code, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)

    def search_examples_by_question(self, question: str, top_k: int):
        question = get_query_question(question)
        with self.lock:
            query_embedding = self.embedder.encode(question, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.example_corpus_by_question_embeddings, top_k=top_k)
            assert len(hits) == 1
            examples = [(self.example[hit['corpus_id']], hit['score']) for hit in hits[0]]
            return examples

    def search_examples_by_sql(self, sql: str, top_k: int):
        sql = get_query_sql(sql)
        with self.lock:
            query_embedding = self.embedder.encode(sql, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.example_corpus_by_sql_embeddings, top_k=top_k)
            assert len(hits) == 1
            examples = [(self.example[hit['corpus_id']], hit['score']) for hit in hits[0]]
            return examples

    def search_examples_by_code(self, code: str, top_k: int):
        with self.lock:
            query_embedding = self.embedder.encode(code, normalize_embeddings=True, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.example_corpus_by_code_embeddings, top_k=top_k)
            assert len(hits) == 1
            examples = [(self.example[hit['corpus_id']], hit['score']) for hit in hits[0]]
            return examples


class Retriever:

    def __init__(self,
                 sbert_model_path: str='./pretrained/bge-large-en-v1.5',
                #  sbert_model_path: str='./pretrained/bge-base-en-v1.5',
                #  sbert_model_path: str='./pretrained/bge-small-en-v1.5',
                 db_set_path: str=DB_SET_PATH,
                 training_set_paths: List[str]=TRAINING_SET_PATHS,
                 db_description_path: Optional[str]=DB_DESCRIPTION_PATH,
        ):
        self.embedder = SentenceTransformer(sbert_model_path)
        # self.lock = Lock()
        self.lock = BoundedSemaphore(50)  # Single GPU Total vRAM (GB) // Single Thread Required vRAM (GB)

        print('loading dbs')
        self.dbs = {}  # db schema by dump_db_json_schema
        self.db_path = {}

        self.col_schema = {}
        self.column_corpus_map = {}
        self.column_corpus = []
        self.column_retriever: Dict[str, SingleDBColumnRetriever] = {}

        self.table_corpus_map = {}
        self.table_corpus = []
        self.table_retriever: Dict[str, SingleDBTableRetriever] = {}

        if db_description_path is not None:
            with open(db_description_path, encoding='utf-8') as f:
                self.descriptions = json.load(f)
        else:
            self.descriptions = {}
        
        for db_path in glob.glob(db_set_path):
            if db_path.endswith('.json'):
                continue
            db_name = db_path.split('/')[-1]
            print(db_path, db_name)
            self.db_path[db_name] = fr'{db_path}/{db_name}.sqlite'
            schema = dump_db_json_schema(fr'{db_path}/{db_name}.sqlite')
            self.dbs[db_name] = schema
            col_schema = []
            with sqlite3.connect(fr'{db_path}/{db_name}.sqlite') as conn:
                cursor = conn.cursor()
                for table in schema:
                    for c, t in table['columns']:
                        comment = self.descriptions.get(db_name, {}).get(table['name'].lower(), {}).get(c.lower(), None)
                        cursor.execute(f'''select distinct "{c}" from "{table['name']}" limit {MAX_VALUES}''')
                        values = [v for v, in cursor.fetchall()]
                        plain_c = plain_column(table['name'], c, t, comment, values)
                        if plain_c not in self.column_corpus_map:
                            self.column_corpus_map[plain_c] = len(self.column_corpus)
                            self.column_corpus.append(plain_c)
                        col_schema.append((table['name'], c, t, comment, values))
                    
                    plain_sql = plain_table_schema(table)
                    if plain_sql not in self.table_corpus_map:
                        self.table_corpus_map[plain_sql] = len(self.table_corpus)
                        self.table_corpus.append(plain_sql)
                cursor.close()
            self.col_schema[db_name] = col_schema
            self.column_retriever[db_name] = SingleDBColumnRetriever(self, col_schema)
            self.table_retriever[db_name] = SingleDBTableRetriever(self, schema, self.descriptions.get(db_name, {}))

        self.column_corpus_embeddings = self.embedder.encode(self.column_corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        self.table_corpus_embeddings = self.embedder.encode(self.table_corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)

        if len(training_set_paths) > 0:
            self.example_retriever = ExampleRetriever(self, training_set_paths)
        else:
            self.example_retriever = None

    def search_examples(self, question: str, top_k: int):
        return self.example_retriever.search_examples_by_question(question, top_k)
    
    def search_examples_by_sql(self, sql: str, top_k: int):
        return self.example_retriever.search_examples_by_sql(sql, top_k)


def generate_sql_zero_shot_chat(client: LLMClient, question: str, schemas: List[Tuple[str, str, str, float]]=None, **generation_configs):
    if schemas is None:
        prompt = f"Write a SQL query to answer the question.\nQuestion: {question}"
    else:
        concat_schemas = '\n'.join([schema for _, _, schema, _ in schemas])
        prompt = f"Write a SQL query to answer the question.\nDatabase Schema:\n{concat_schemas}\n\nQuestion: {question}"
    # print(prompt)
    # exit()
    res, usage = client.chat(
        messages=[{
            'role': 'user',
            'content': prompt,
        }],
        **generation_configs,
    )
    return res, usage


def is_safe_char(c: str):
    import string
    return c in string.ascii_letters + string.digits + '_'


def is_safe_first_char(c: str):
    import string
    return c in string.ascii_letters + '_'


def convert_to_safe_var_name(s: str):
    import string
    assert len(s) > 0
    res = []
    for i, c in enumerate(s):
        if i == 0:
            if is_safe_first_char(c):
                res.append(c)
            else:
                res.append('_')
                if c in string.digits:
                    res.append(c)
        else:
            if is_safe_char(c):
                res.append(c)
            else:
                res.append('_')
    return ''.join(res)


def schema_post_process(schema):
    if BACKEND_DB != DB_ORACLE:
        return schema

    def replace_table_name(match_obj):
        return f"__tablename__ = quoted_name('{match_obj.group(1)}', quote=True)"

    schema = re.sub(r"__tablename__ = '(.*?)'", replace_table_name, schema)

    def replace_column_name(match_obj):
        return f" = mapped_column(quoted_name('{match_obj.group(1)}', quote=True)"

    schema = re.sub(r" = mapped_column\('(.*?)'", replace_column_name, schema)

    return schema


def get_sqlalchemy_code(schemas: List[Tuple[str, str, str, float]], db_name: str=None):
    prompt = """from typing import List
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy import func
from sqlalchemy import *
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.orm import aliased
import datetime
import decimal


class Base(DeclarativeBase):
    pass

"""
    if db_name is not None:
        concat_schemas = '\n'.join([schema_post_process(schema) for _, _, schema, _ in schemas])
    else:
        concat_schemas = '\n'.join([schema for _, _, schema, _ in schemas])
    prompt += concat_schemas + '\n'
    if db_name is not None:
        if BACKEND_DB == DB_SQLITE:
            prompt += 'engine = create_engine("sqlite://", echo=True)\n'
        elif BACKEND_DB == DB_PG:
            conn_str = f"postgresql+psycopg2://{db_conn_config['user']}:{db_conn_config['password']}@{db_conn_config['host']}:{db_conn_config['port']}/{db_name}"
            prompt += f'engine = create_engine("{conn_str}", echo=True)\n'
            prompt += 'engine = engine.execution_options(postgresql_readonly=True)\n'
        elif BACKEND_DB == DB_MSSQL:
            conn_str = f"mssql+pyodbc://{db_conn_config['user']}:{db_conn_config['password']}@{db_conn_config['host']}:{db_conn_config['port']}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
            prompt += f'engine = create_engine("{conn_str}", echo=True)\n'
        elif BACKEND_DB == DB_ORACLE:
            conn_str = f"oracle+oracledb://{db_conn_config['user']}:{db_conn_config['password']}@{db_conn_config['host']}:{db_conn_config['port']}?service_name={db_name}"
            prompt += f'engine = create_engine("{conn_str}", echo=True)\n'
        elif BACKEND_DB == DB_MYSQL:
            conn_str = f"mysql+pymysql://{db_conn_config['user']}:{db_conn_config['password']}@{db_conn_config['host']}:{db_conn_config['port']}/{db_name}"
            prompt += f'engine = create_engine("{conn_str}", echo=True)\n'
        else:
            raise ValueError(f'invalid BACKEND_DB: {BACKEND_DB}')
    else:
        prompt += 'engine = create_engine("sqlite://", echo=True)\n'
    return prompt


def sql_post_process(sql):
    if BACKEND_DB == DB_PG:

        def convert_strftime(sql):

            def replace(match_obj):
                format_str = match_obj.group(1)
                column = match_obj.group(2).strip()
                if format_str == '%Y':
                    return f"to_char({column}::date, 'YYYY')"
                elif format_str == '%m':
                    return f"to_char({column}::date, 'MM')"
                elif format_str == '%Y-%m':
                    return f"to_char({column}::date, 'YYYY-MM')"
                elif format_str == '%Y-%m-%d':
                    return f"to_char({column}::date, 'YYYY-MM-DD')"
                elif format_str == '%H:%M:%S':
                    return f"to_char({column}::time, 'HH24:MI:SS')"
                else:
                    return match_obj.group(0)
                
            if 'strftime(' in sql.lower():
                sql = re.sub(r"strftime\('(.*?)',(.*?)\)", replace, sql, flags=re.IGNORECASE)
            return sql

        sql = sql.replace('%%', '%')
        sql = sql.replace(" DESC ", " DESC NULLS LAST ")
        sql = sql.replace(" LIKE ", " ILIKE ")
        sql = convert_strftime(sql)
    elif BACKEND_DB == DB_MSSQL:

        def convert_strftime(sql):

            def replace(match_obj):
                format_str = match_obj.group(1)
                column = match_obj.group(2).strip()
                if format_str == '%Y':
                    return f"format(cast({column} as date), 'yyyy')"
                elif format_str == '%m':
                    return f"format(cast({column} as date), 'MM')"
                elif format_str == '%Y-%m':
                    return f"format(cast({column} as date), 'yyyy-MM')"
                elif format_str == '%Y-%m-%d':
                    return f"format(cast({column} as date), 'yyyy-MM-dd')"
                elif format_str == '%H:%M:%S':
                    return f"format(cast({column} as datetime), 'HH:mm:ss')"
                else:
                    return match_obj.group(0)
                
            if 'strftime(' in sql.lower():
                sql = re.sub(r"strftime\('(.*?)',(.*?)\)", replace, sql, flags=re.IGNORECASE)
            return sql

        def cast_avg(sql):

            def replace(match_obj):
                column = match_obj.group(1)
                return f"avg(cast({column} as real))"
                
            if 'avg(' in sql.lower():
                sql = re.sub(r"avg\((.*?)\)", replace, sql, flags=re.IGNORECASE)
            return sql

        def convert_ilike(sql):
            
            def replace(match_obj):
                import string

                format_str = match_obj.group(1)
                need_replace = False
                for c in format_str:
                    if c in string.ascii_letters:
                        need_replace = True
                        break
                if need_replace:
                    return f" COLLATE SQL_Latin1_General_CP1_CI_AS LIKE '{format_str}'"
                else:
                    return match_obj.group(0)
                
            if ' like ' in sql.lower():
                sql = re.sub(r" like '(.*?)'", replace, sql, flags=re.IGNORECASE)
            return sql

        sql = sql.replace(" substr(", " substring(")
        sql = cast_avg(sql)
        sql = convert_ilike(sql)
        sql = convert_strftime(sql)
    elif BACKEND_DB == DB_ORACLE:

        def convert_strftime(sql):

            def replace(match_obj):
                format_str = match_obj.group(1)
                column = match_obj.group(2).strip()
                if format_str == '%Y':
                    return f"to_char(cast({column} as date), 'YYYY')"
                elif format_str == '%m':
                    return f"to_char(cast({column} as date), 'MM')"
                elif format_str == '%Y-%m':
                    return f"to_char(cast({column} as date), 'YYYY-MM')"
                elif format_str == '%Y-%m-%d':
                    return f"to_char(cast({column} as date), 'YYYY-MM-DD')"
                elif format_str == '%H:%M:%S':
                    return f"to_char(cast({column} as timestamp), 'HH24:MI:SS')"
                else:
                    return match_obj.group(0)
                
            if 'strftime(' in sql.lower():
                sql = re.sub(r"strftime\('(.*?)',(.*?)\)", replace, sql, flags=re.IGNORECASE)
            return sql

        sql = sql.replace(" DESC\n", " DESC NULLS LAST\n")
        sql = convert_strftime(sql)
    elif BACKEND_DB == DB_MYSQL:

        def convert_strftime(sql):

            def replace(match_obj):
                format_str = match_obj.group(1)
                column = match_obj.group(2).strip()
                if format_str == '%Y':
                    return f"date_format(cast({column} as date), '%Y')"
                elif format_str == '%m':
                    return f"date_format(cast({column} as date), '%m')"
                elif format_str == '%Y-%m':
                    return f"date_format(cast({column} as date), '%Y-%m')"
                elif format_str == '%Y-%m-%d':
                    return f"date_format(cast({column} as date), '%Y-%m-%d')"
                elif format_str == '%H:%M:%S':
                    return f"date_format(cast({column} as time), '%H:%i:%S')"
                else:
                    return match_obj.group(0)
                
            if 'strftime(' in sql.lower():
                sql = re.sub(r"strftime\('(.*?)',(.*?)\)", replace, sql, flags=re.IGNORECASE)
            return sql

        def convert_ilike(sql):
            
            def replace(match_obj):
                import string

                format_str = match_obj.group(1)
                need_replace = False
                for c in format_str:
                    if c in string.ascii_letters:
                        need_replace = True
                        break
                if need_replace:
                    return f" COLLATE utf8mb4_0900_ai_ci LIKE '{format_str}'"
                else:
                    return match_obj.group(0)
                
            if ' like ' in sql.lower():
                sql = re.sub(r" like '(.*?)'", replace, sql, flags=re.IGNORECASE)
            return sql

        sql = sql.replace('%%', '%')
        # sql = sql.replace(" DESC ", " DESC NULLS LAST ")
        sql = convert_ilike(sql)
        sql = convert_strftime(sql)
    return sql


@timeout_after(10)
def convert_sqlalchemy_to_sql(code_prefix: str, stmts: str, check_for_error_only=False) -> str | None:
    code = code_prefix
    code += 'with Session(engine) as unused_session:\n'
    code += stmts.rstrip() + '\n'
    if not check_for_error_only:
        code += '    stmt = str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))'
    if DEBUG:
        print('=' * 10, 'CONVERT', '=' * 10)
        print(code[code.find('with'):])
        # print(code)
        print('=' * 29)
    vars = {}
    exec(code, vars)
    if not check_for_error_only:
        sql = vars['stmt']
        return sql


def generate_plan_by_decomposing_with_sqlalchemy_chat(client: LLMClient, db_name: str, question: str, schemas: List[Tuple[str, str, str, float]], examples: List[Tuple[dict, float]]=None, gold_sql: str=None, **generation_configs):
    code_prefix_full = get_sqlalchemy_code(schemas, db_name)
    code_prefix = get_sqlalchemy_code(schemas)
    prompt = "Complete the following code in Python:\n```python\n"
    prompt += code_prefix
    prompt += 'with Session(engine) as session:\n'
    prompt += f"\n"
    
    if examples is not None and len(examples) > 0:
        prompt += f'    """\n'
        prompt += f'    # Here are some examples for reference:\n'
        for e, s in examples:
            prompt += f"\n"
            prompt += f"    # Question: {e['nl']}\n"
            # prompt += f'    """\n'
            if 'code' in e:
                prompt += f"{e['code']}\n"
            else:
                prompt += f"{e['sql']}\n"
            # prompt += f'    """\n\n'
        prompt += f'    """\n\n'
    
    prompt += f"    # Question: {question}\n"
    if gold_sql is not None:
        assert 'train' in DEV_SET_PATH, 'THIS IS NOT THE TRAINING SET! WHERE DO YOU GET THE GOLD SQL?'
        prompt += f"    # SQL: {gold_sql}\n"
        
    def generate_code(prompt: str, add_begin_of_code_block=True):
        if add_begin_of_code_block:
            # prompt += f'    """\n'
            prompt += f'    {{Your Code Here}}\n\n'
            # prompt += f'    """\n'
            prompt += f'    result = session.execute(stmt)\n'
            prompt += f'    print(result)\n'
            prompt += f'```\n\n'
            prompt += "Your response should strictly follow this format:\n"
            prompt += "```python\n"
            prompt += "{Your Code}\n"
            prompt += "```\n\n"
            prompt += "Your code should only contain the query statements--no input, output, or execution statements."
        if 'n' in generation_configs:
            assert generation_configs['n'] == 1
        res, usage = client.chat(
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            # stop=['\n\n'],
            max_tokens=2048,
            **generation_configs,
        )
        assert len(res) == 1
        res: str = res[0]
        if DEBUG:
            print('=' * 10, 'IN', '=' * 10)
            # print(prompt[prompt.rfind('    # Question: '):])
            # print(prompt[prompt.find('with'):])
            print(prompt)
            print('=' * 10, 'OUT', '=' * 10)
            print(res)
            print('=' * 50)
        if res.rfind('```python\n') != -1:
            res = res[res.rfind('```python\n')+len('```python\n'):res.rfind('\n```')]
            # ensure one indent
            lines = [line for line in res.split('\n') if line != '']
            if not lines[0].startswith('    '):
                res = '    ' + res.replace('\n', '\n    ')
            if DEBUG:
                print('*' * 10, 'FORMAT', '*' * 10)
                print(res)
                print('*' * 50)
            end_of_code_block = True
        else:
            end_of_code_block = False
        return end_of_code_block, usage, res

    rounds = []

    total_usage = None
    num_rounds = 0
    while num_rounds < MAX_ROUNDS:
        num_rounds += 1
        end_of_code_block, usage, res = generate_code(prompt)
        round_res = {
            'code': res,
            'usage': usage,
        }
        if total_usage is None:
            total_usage = deepcopy(usage)
        else:
            add_usage_inplace(total_usage, usage)
        if end_of_code_block:
            try:
                sql = convert_sqlalchemy_to_sql(code_prefix_full, res)
                sql = sql_post_process(sql)
                error = check_sql_executability(sql, db_name)
                if error is not None:
                    message = str(error)
                    raise ValueError('SQL: ' + sql + '\nError: ' + error)
                else:
                    message = 'OK'

                round_res['infer'] = sql
                round_res['message'] = message
                rounds.append(round_res)

                break
            except Exception as e:
                # print(e)
                sql = 'SELECT'
                message = str(e)
        else:
            # print('code block!')
            sql = 'SELECT'
            message = 'end of code block is missing'
        
        round_res['infer'] = sql
        round_res['message'] = message
        rounds.append(round_res)

    return res, sql, message, total_usage, rounds


def convert_db_type_to_python_type(db_type: str):
    if db_type.lower() in ['real', 'float', 'double']:
        return 'float'
    elif db_type.lower() in ['integer', 'int', 'bigint', 'smallint', 'smallint unsigned', 'tinyint unsigned', 'mediumint unsigned']:
        return 'int'
    elif db_type.lower() in ['date', 'year']:
        return 'datetime.date'
    elif db_type.lower() in ['datetime', 'timestamp']:
        return 'datetime.datetime'
    elif db_type.lower().startswith('numeric') or db_type.lower().startswith('decimal') or db_type.lower().startswith('number') \
            or db_type.lower().startswith('float(') or db_type.lower().startswith('int(') or db_type.lower().startswith('bigint('):
        return 'decimal.Decimal'
    elif db_type.lower() in ['boolean', 'bool']:
        return 'bool'
    return 'str'


def build_schema(table_name, columns, primary_keys, foreign_keys, descriptions=None):
    if descriptions is None:
        descriptions = {}
    if len(primary_keys) == 0:
        def check_exists(name):
            for c, _ in columns:
                if c.lower() == name:
                    return True
            return False
        fake_pk_name = 'id'
        if check_exists(fake_pk_name):
            i = 0
            while check_exists(f'id{i}'):
                i += 1
            fake_pk_name = f'id{i}'
        assert not check_exists(fake_pk_name)
        primary_keys = primary_keys + [fake_pk_name]
        columns = [(fake_pk_name, 'TEXT')] + columns
        descriptions = {fake_pk_name: 'DO NOT USE THIS COLUMN!', **descriptions}
    columns = [[c, t, False, None] for c, t in columns]
    for pk in primary_keys:
        for c in columns:
            if c[0].lower() == pk.lower():
                c[2] = True
                break
    for fk in foreign_keys:
        if len(fk['fk']) == 1 and fk['ref_key'][0] is not None:
            for c in columns:
                if c[0].lower() == fk['fk'][0].lower():
                    c[3] = f"`{fk['ref']}`.`{fk['ref_key'][0]}`"
                    break
    schema = f"""class {convert_to_safe_var_name(table_name)}(Base):
    __tablename__ = {repr(table_name)}
"""
    for c, t, pk, fk in columns:
        if pk:
            if fk:
                schema += f'    {convert_to_safe_var_name(c)}: Mapped[{convert_db_type_to_python_type(t)}] = mapped_column({repr(c)}, ForeignKey({repr(fk)}), primary_key=True)'
            else:
                schema += f'    {convert_to_safe_var_name(c)}: Mapped[{convert_db_type_to_python_type(t)}] = mapped_column({repr(c)}, primary_key=True)'
        else:
            if fk:
                schema += f'    {convert_to_safe_var_name(c)}: Mapped[Optional[{convert_db_type_to_python_type(t)}]] = mapped_column({repr(c)}, ForeignKey({repr(fk)}))'
            else:
                schema += f'    {convert_to_safe_var_name(c)}: Mapped[Optional[{convert_db_type_to_python_type(t)}]] = mapped_column({repr(c)})'
        if descriptions.get(c.lower(), None):
            schema += '  # ' + descriptions[c.lower()]
        schema += '\n'
    return schema


def convert_retrieved_columns_to_table_schemas(retriever: Retriever, r: dict, retrieved_columns, enable_comment=True, enable_values=True):
    pks = {}
    pks_orig = {}
    fks = {}
    for table in retriever.dbs[get_database_name(r['db_id'])]:
        pks[table['name'].lower()] = [pk.lower() for pk in table['primary_keys']]
        pks_orig[table['name']] = table['primary_keys']
        fks[table['name'].lower()] = [{
            'fk': [k.lower() for k in fk['fk']],
            'ref': fk['ref'].lower(),
            'ref_key': [k.lower() for k in fk['ref_key']],
            'original_fk': fk,
        } for fk in table['foreign_keys']
        if all([k is not None for k in fk['fk']])
            and all([k is not None for k in fk['ref_key']])]

    columns = {}
    for table_name, column_name, column_type, comment, values in retriever.col_schema[get_database_name(r['db_id'])]:
        if table_name.lower() not in columns:
            columns[table_name.lower()] = {}
        assert column_name.lower() not in columns[table_name.lower()]
        columns[table_name.lower()][column_name.lower()] = (
            (column_name, column_type),
            # f"{comment} Examples: {', '.join([repr(v) for v in values])}"
            format_comment(comment, values, enable_comment, enable_values)
        )

    schemas = {}
    descriptions = {}
    for _, _, (table_name, column_name, column_type, comment, values), _ in retrieved_columns:
        if table_name not in schemas:
            schemas[table_name] = []
            descriptions[table_name] = {}
        schemas[table_name].append((column_name, column_type))
        if column_name is not None:
            descriptions[table_name][column_name.lower()] = format_comment(comment, values, enable_comment, enable_values)  # f"{comment} Examples: {', '.join([repr(v) for v in values])}"
    
    schema_columns = {
        t.lower(): set([c.lower() for c, _ in schemas[t] if c is not None])
        for t in schemas
    }

    def add_column_to_table_if_not_exists(t, k):
        if k not in schema_columns[t.lower()]:
            c, d = columns[t.lower()][k]
            schemas[t].append(c)
            descriptions[t][k] = d
            schema_columns[t.lower()].add(k)
    
    def is_valid_column(t, k):
        return t.lower() in columns and k in columns[t.lower()]

    valid_fks = {}
    for t in schemas:
        for k in pks[t.lower()]:
            add_column_to_table_if_not_exists(t, k)
        valid_fks[t] = []
        for fk in fks[t.lower()]:
            if fk['ref'] in schema_columns:
                ref_table = None
                for tn in schemas:
                    if tn.lower() == fk['ref']:
                        ref_table = tn
                        break
                assert ref_table is not None
                is_valid_fk = True
                for k in fk['fk']:
                    if not is_valid_column(t, k):
                        is_valid_fk = False
                        break
                for k in fk['ref_key']:
                    if not is_valid_column(ref_table, k):
                        is_valid_fk = False
                        break
                if is_valid_fk:
                    for k in fk['fk']:
                        add_column_to_table_if_not_exists(t, k)
                    for k in fk['ref_key']:
                        add_column_to_table_if_not_exists(ref_table, k)
                    valid_fks[t].append(fk['original_fk'])

    if METHOD in ['sqlalchemy']:
        retrieved_schemas = [(t, None, build_schema(t, [cc for cc in c if cc != (None, None)], pks_orig[t], valid_fks[t], descriptions[t]), 1.0) for t, c in schemas.items()]
    else:
        retrieved_schemas = [(t, None, build_sql(t, [cc for cc in c if cc != (None, None)], pks_orig[t], valid_fks[t], descriptions[t]), 1.0) for t, c in schemas.items()]
    
    return retrieved_schemas


# a += b
def add_usage_inplace(a: dict, b: Optional[dict]):
    assert a is not None
    if b is None:
        return
    for k in a:
        if k in b:
            if isinstance(a[k], dict):
                add_usage_inplace(a[k], b[k])
            else:
                if a[k] is None:
                    a[k] = b[k]
                elif b[k] is None:
                    pass
                else:
                    a[k] += b[k]
    for k in b:
        if k not in a:
            a[k] = b[k]


def get_gold_columns(r: dict):
    # gold column
    infer_gold_schemas = get_tables_and_columns(r['query'])
    gold_columns = [
        (
            f'"{tname}"."{cname}"',
            None,
            (tname, cname, ctype, c, v),
            1.0
        )
        for tname, cname, ctype, c, v in retriever.column_retriever[get_database_name(r['db_id'])].corpus_full
        if tname.lower() in infer_gold_schemas
            and cname.lower() in infer_gold_schemas[tname.lower()]
    ]

    # what if tables w/o columns?
    gold_columns += list(set([
        (
            f'"{tname}".*',
            None,
            (tname, None, None, None, None),
            1.0
        )
        for tname, cname, ctype, c, v in retriever.column_retriever[get_database_name(r['db_id'])].corpus_full
        if tname.lower() in infer_gold_schemas
            and len(infer_gold_schemas[tname.lower()]) == 0
    ]))

    return infer_gold_schemas, gold_columns


def process_record_sqlalchemy_full_db(client: LLMClient, retriever: Retriever, r: dict, mode: str):
    if FEW_SHOT_K > 0:
        examples = retriever.search_examples(r['question'], FEW_SHOT_K)
    else:
        examples = []

    full_columns = [
        (
            f'"{tname}"."{cname}"',
            None,
            (tname, cname, ctype, c, v),
            1.0
        )
        for tname, cname, ctype, c, v in retriever.column_retriever[get_database_name(r['db_id'])].corpus_full
    ]
    full_schemas = convert_retrieved_columns_to_table_schemas(retriever, r, full_columns)

    if WITH_GOLD_SQL:
        gold_sql = r['query']

        try:
            infer_gold_schemas, gold_columns = get_gold_columns(r)
            gold_schemas = convert_retrieved_columns_to_table_schemas(retriever, r, gold_columns)
            concat_schemas = gold_schemas
        except:
            gold_schemas = None
            concat_schemas = full_schemas
    else:
        gold_sql = None
        gold_schemas = None
        concat_schemas = full_schemas

    if mode == FEW_SHOT_COMPLETION:
        assert False, 'not supported'
    elif mode == ZERO_SHOT_CHAT:
        if BACKEND_DB == DB_SQLITE:
            db_name_or_path = retriever.db_path[get_database_name(r['db_id'])]
        else:
            db_name_or_path = get_database_name(r['db_id'])
        code, infer, message, usage, rounds = generate_plan_by_decomposing_with_sqlalchemy_chat(client, db_name_or_path, r['question'], concat_schemas, examples, gold_sql)
    else:
        raise ValueError(f'invalid mode: {mode}')

    return {
        'id': r['id'],
        'question': r['question'],
        # 'output': r['query'],
        'db': r['db_id'],
        'fs_examples': examples,
        # 'full_columns': full_columns,
        'full_schemas': full_schemas,
        'gold_schemas': gold_schemas,
        'rounds': rounds,
        'code': code,
        'infer': infer,
        'message': message,
        # 'result_var': 'result',
        'usage': usage,
    }


def process_record(client: LLMClient, retriever: Retriever, r: dict):
    if GEN_MODE == SQL_MODE:
        if 'infer' in r:
            return r
        gold_sql = r['query']
        if not WITH_GOLD_SQL:
            r.pop('query')
        if METHOD == 'sqlalchemy':
            func = process_record_sqlalchemy_full_db
        else:
            raise ValueError(f'invalid METHOD: {METHOD}')
    else:
        raise ValueError(f'unknown GEN_MODE={GEN_MODE}')
    
    if DEBUG:
        res = func(client, retriever, r, MODE)
    else:
        try:
            res = func(client, retriever, r, MODE)
        except Exception as e:
            print('error:', e)
            return None
    
    if GEN_MODE == SQL_MODE:
        r['query'] = res['output'] = gold_sql
    return res


if __name__ == '__main__':
    # local
    llm_config = dict(host='127.0.0.1', port=8080)
    # gpt
    # llm_config = dict(
    #         api_key="sk-...",
    #         base_url="https://xxx.yyy.zzz/v1",
    #         model="gpt-4o-2024-11-20",
    # )
    client = LLMClient(**llm_config)
    retriever = Retriever()

    if GEN_MODE == SQL_MODE:
        with open(DEV_SET_PATH, encoding='utf-8') as f:
            data = json.load(f)
            for i, r in enumerate(data):
                r['id'] = i
                if DB_NAME in ['bird', 'bird-train']:
                    r['original_question'] = r['question']
                    r['question'] += ' Hint: ' + r['evidence']
                    r['query'] = r['SQL']
    else:
        raise ValueError(f'unknown GEN_MODE={GEN_MODE}')

    if GEN_MODE == SQL_MODE:
        out_file = f'output/{client.model}-{DB_NAME}-{get_file_suffix()}.json'
    else:
        raise ValueError(f'unknown GEN_MODE={GEN_MODE}')

    if DEBUG:
        data = data[0:10]
    # else:
    #     data = data[0:10]
    
    if not DEBUG and os.path.exists(out_file):
        print('loaded:', out_file)
        with open(out_file, encoding='utf-8') as f:
            prev_data = json.load(f)
        assert len(data) == len(prev_data)
        new_data = [
            r if prev_r is None else prev_r
            for r, prev_r in zip(data, prev_data)
        ]
    else:
        new_data = data

    start = time.time()

    if DEBUG:
        max_workers = 1
    else:
        max_workers = 50

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        res_data = list(tqdm(executor.map(lambda r: process_record(client, retriever, r), new_data), total=len(new_data)))

    end = time.time()
    print('elapsed:', end - start, 'seconds')

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res_data, f, ensure_ascii=False, indent=4)
    print('saved:', out_file)

    assert len(data) == len(res_data)
    for i, o in zip(data, res_data):
        if o is None:
            continue
        assert i['id'] == o['id']
        assert i['question'] == o['question']
        assert i['query'] == o['output']
        if DB_NAME in ['bird', 'bird-train']:
            assert i['SQL'] == o['output']
        assert i['db_id'] == o['db']
        # o.pop('id')