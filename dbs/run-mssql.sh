# ensure you are in the dbs directory!
# mkdir bird_mssql_db
# chmod 777 bird_mssql_db
docker run --rm -it -d \
   -e 'ACCEPT_EULA=Y' \
   -e 'MSSQL_SA_PASSWORD=YOUR_PASSWORD' \
   --name 'mssql1' \
   -p 1433:1433 \
   -v $(pwd)/bird_mssql_db:/var/opt/mssql \
   mcr.microsoft.com/mssql/server:2022-latest
