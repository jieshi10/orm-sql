import glob
from sql_utils_2 import dump_db_json_schema, set_identifier, build_sql
import pyodbc
import sqlite3


set_identifier('"')

conn_config = dict(
    dsn="MSSQLServerDatabase", uid="sa", pwd="YOUR_PASSWORD"
)


root_path = 'benchmark/BIRD/dev/dev_databases/*'
# root_path = 'benchmark/BIRD/train/train_databases/*'
# root_path = 'benchmark/spider/database/*'


all_types = set()

conn = pyodbc.connect(f'DSN={conn_config["dsn"]};UID={conn_config["uid"]};PWD={conn_config["pwd"]}')
conn.autocommit=True

for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)

    with conn.cursor() as cursor:
        cursor.execute(f'create database {db_name} collate SQL_Latin1_General_CP1_CS_AS')
    
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
    if db_type.lower() in ['text']:
        return 'nvarchar(max)'
    return db_type


for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    schema = dump_db_json_schema(f'{db_path}/{db_name}.sqlite')
    with pyodbc.connect(f'DSN={conn_config["dsn"]};UID={conn_config["uid"]};PWD={conn_config["pwd"]};DATABASE={db_name}') as conn, \
            sqlite3.connect(f'file:{db_path}/{db_name}.sqlite?mode=ro', uri=True) as conn0:
        conn.autocommit=False
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
            print(build_sql(t['name'], t['columns'], [], []))

            # 1.1: load data from sqlite
            cursor = conn0.cursor()
            cursor.execute('select * from "' + t['name'] + '"')
            data = cursor.fetchall()
            cursor.close()
            print(data[:3])

            with conn.cursor() as cursor:
                # cursor.fast_executemany = True

                # cursor.setinputsizes([(pyodbc.SQL_WVARCHAR, 0, 0)])

                # 1.2: create table in pg
                cursor.execute(build_sql(t['name'], t['columns'], [], []))
                
                # 1.3: insert data into pg
                if len(data) > 0:
                    cursor.executemany('insert into "' + t['name'] + '" values (' + ', '.join(['?'] * len(data[0])) + ')', data)
            
            conn.commit()

        # step 2 (optional): add foreign keys
        # break


conn = pyodbc.connect(f'DSN={conn_config["dsn"]};UID={conn_config["uid"]};PWD={conn_config["pwd"]}')
conn.autocommit=True

for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)

    with conn.cursor() as cursor:
        cursor.execute(f'alter database {db_name} set read_only')

conn.close()