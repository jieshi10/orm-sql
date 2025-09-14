# ensure you are in the dbs directory!
# mkdir bird_oracle_db
# chmod 777 bird_oracle_db
docker run --rm -it -d \
    --name oracledb1 \
    -p 1521:1521 -p 5500:5500 -p 2484:2484 \
    -e ORACLE_PWD=YOUR_PASSWORD \
    -e ENABLE_ARCHIVELOG=true \
    -e ENABLE_FORCE_LOGGING=true \
    -e ENABLE_TCPS=true \
    -v $(pwd)/bird_oracle_db:/opt/oracle/oradata \
    oracle/database:23.6.0-free