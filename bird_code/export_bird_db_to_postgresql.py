import glob
from sql_utils_2 import dump_db_json_schema, set_identifier, build_sql
import psycopg2
from psycopg2.extras import execute_batch
import sqlite3


set_identifier('"')

conn_config = dict(
    host="127.0.0.1", port=5432, user="postgres", password="YOUR_PASSWORD"
)


root_path = 'benchmark/BIRD/dev/dev_databases/*'
# root_path = 'benchmark/BIRD/train/train_databases/*'
# root_path = 'benchmark/spider/database/*'


all_types = set()

conn = psycopg2.connect(dbname="postgres", **conn_config)
conn.set_session(autocommit=True)

for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)

    with conn.cursor() as cursor:
        cursor.execute(f'create database {db_name}')
    
    schema = dump_db_json_schema(f'{db_path}/{db_name}.sqlite')
    for t in schema:
        if t['name'] == 'sqlite_sequence':
            continue
        all_types |= set([tt for c, tt in t['columns']])
        for c, tt in t['columns']:
            if tt == '':
                print(t['name'], c)
        # for fk in t['foreign_keys']:
        #     print(fk)
        # print(t['sql'])
        # print(build_sql(t['name'], t['columns'], t['primary_keys'], t['foreign_keys']))
    # break

conn.close()

print(all_types)


def convert_sqlite_type_to_postgres_type(db_type: str):
    if db_type.lower() in ['datetime']:
        return 'timestamp'
    return db_type


for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    schema = dump_db_json_schema(f'{db_path}/{db_name}.sqlite')
    with psycopg2.connect(dbname=db_name, **conn_config) as conn, \
            sqlite3.connect(f'file:{db_path}/{db_name}.sqlite?mode=ro', uri=True) as conn0:
        conn.set_session(autocommit=False)
        # step 1: create table & insert data
        for t in schema:
            if t['name'] == 'sqlite_sequence':
                continue
            for c, tt in t['columns']:
                if tt == '':
                    print(t['name'], c)
            for fk in t['foreign_keys']:
                print(fk)
            t['columns'] = [(c, convert_sqlite_type_to_postgres_type(tt)) for c, tt in t['columns']]
            print(t['columns'])
            # print(t['sql'])
            # print(build_sql(t['name'], t['columns'], t['primary_keys'], t['foreign_keys']))
            print(build_sql(t['name'], t['columns'], t['primary_keys'], []))

            # 1.1: load data from sqlite
            cursor = conn0.cursor()
            cursor.execute('select * from "' + t['name'] + '"')
            data = cursor.fetchall()
            cursor.close()
            print(data[:3])

            with conn.cursor() as cursor:
                # 1.2: create table in pg
                cursor.execute(build_sql(t['name'], t['columns'], t['primary_keys'], []))
                
                # 1.3: insert data into pg
                if len(data) > 0:
                    execute_batch(cursor, 'insert into "' + t['name'] + '" values (' + ', '.join(['%s'] * len(data[0])) + ')', data)
            
            conn.commit()

        # step 2 (optional): add foreign keys
        # break
