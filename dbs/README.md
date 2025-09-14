# Setting up Databases

This doc briefly introduces the necessary steps to start the databases.

The working directory should be `dbs` (unless otherwise explicitly specified).

## PostgreSQL
- Run:
  ```shell
  . run-postgresql.sh
  ```

## MySQL
- Run:
  ```shell
  . run-mysql.sh
  ```

## SQL Server
- Run the following commands for the first time:
  ```shell
  mkdir bird_mssql_db
  chmod 777 bird_mssql_db
  ```
- Run:
  ```shell
  . run-mssql.sh
  ```
- Follow this [link](https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-SQL-Server-from-Linux) to install the pyodbc driver for connecting to SQL Server. The key steps are:
  ```shell
  apt update && apt install -y curl
  curl -sSL -O https://packages.microsoft.com/config/ubuntu/$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2)/packages-microsoft-prod.deb
  dpkg -i packages-microsoft-prod.deb
  apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18
  echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
  apt-get install -y unixodbc-dev
  odbcinst -i -s -f dsn.ini -l
  ```

## Oracle
- Refer to this [link](https://github.com/oracle/docker-images/blob/main/OracleDatabase/SingleInstance/README.md) to build the Oracle docker image.
- Run the following commands for the first time:
  ```shell
  mkdir bird_oracle_db
  chmod 777 bird_oracle_db
  ```
- Run:
  ```shell
  . run-oracle.sh
  ```
