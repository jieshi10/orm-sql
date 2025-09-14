import glob
from sql_utils_2 import dump_db_json_schema, set_identifier, build_sql
import oracledb
import sqlite3


set_identifier('"')

conn_config = dict(
    host="127.0.0.1", port=1521, user="test_user", password="YOUR_PASSWORD"
)


root_path = 'benchmark/BIRD/dev/dev_databases/*'
# root_path = 'benchmark/BIRD/train/train_databases/*'
# root_path = 'benchmark/spider/database/*'


all_types = set()

conn = oracledb.connect(user='sys', password=conn_config['password'], host=conn_config['host'], port=conn_config['port'], service_name='FREE', mode=oracledb.AUTH_MODE_SYSDBA)
conn.autocommit=True

for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)

    with conn.cursor() as cursor:
        cursor.execute(f'''create pluggable database {db_name} admin user {conn_config["user"]} identified by "{conn_config["password"]}" FILE_NAME_CONVERT = ('/opt/oracle/oradata/FREE/pdbseed/', '/opt/oracle/oradata/FREE/{db_name}/')''')
        cursor.execute(f'''alter pluggable database {db_name} open''')
    
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

for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)

    conn = oracledb.connect(user='sys', password=conn_config['password'], host=conn_config['host'], port=conn_config['port'], service_name=db_name, mode=oracledb.AUTH_MODE_SYSDBA)
    conn.autocommit=True

    with conn.cursor() as cursor:
        cursor.execute(f'''grant resource to test_user''')
        cursor.execute(f'''alter user test_user quota unlimited on system''')

    conn.close()


def convert_sqlite_type_to_postgres_type(db_type: str):
    if db_type.lower() in ['text']:
        return 'nvarchar2(1333)'
    elif db_type.lower() in ['datetime']:
        return 'timestamp'
    return db_type


print('exporting')


for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)
    schema = dump_db_json_schema(f'{db_path}/{db_name}.sqlite')
    with oracledb.connect(user=conn_config['user'], password=conn_config['password'], host=conn_config['host'], port=conn_config['port'], service_name=db_name) as conn, \
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
            print(build_sql(t['name'], t['columns'], [], [])[:-1])

            # 1.1: load data from sqlite
            cursor = conn0.cursor()
            cursor.execute('select * from "' + t['name'] + '"')
            data = cursor.fetchall()
            cursor.close()
            print(data[:3])

            with conn.cursor() as cursor:
                # cursor.fast_executemany = True

                # cursor.setinputsizes([(pyodbc.SQL_WVARCHAR, 0, 0)])

                cursor.execute("alter session set NLS_DATE_FORMAT='YYYY-MM-DD'")
                cursor.execute("alter session set NLS_TIMESTAMP_FORMAT='YYYY-MM-DD HH24:MI:SSXFF'")

                # 1.2: create table in pg
                cursor.execute(build_sql(t['name'], t['columns'], [], [])[:-1])
                
                # 1.3: insert data into pg
                if len(data) > 0:
                    cursor.executemany('insert into "' + t['name'] + '" values (' + ', '.join([f':{i}' for i in range(len(data[0]))]) + ')', [tuple([(col[:1333] if isinstance(col, str) else col) for col in row]) for row in data])
            
            conn.commit()

        # step 2 (optional): add foreign keys
        # break


print('altering db state')


for db_path in glob.glob(root_path):
    if db_path.endswith('.json'):
        continue
    db_name = db_path.split('/')[-1]
    print(db_path, db_name)

    conn = oracledb.connect(user='sys', password=conn_config['password'], host=conn_config['host'], port=conn_config['port'], service_name=db_name, mode=oracledb.AUTH_MODE_SYSDBA)
    conn.autocommit=True

    with conn.cursor() as cursor:
        cursor.execute(f'alter pluggable database {db_name} close')
        try:
            cursor.execute(f'alter pluggable database {db_name} open read only')
        except Exception:
            pass

    conn.close()

    # break