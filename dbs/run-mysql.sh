docker run --rm -it -d \
    --name mysql1 \
    -v $(pwd)/bird_mysql_db/:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=YOUR_PASSWORD \
    -p 3306:3306 \
    mysql:9.3 \
    --character-set-server=utf8mb4 \
    --collation-server=utf8mb4_0900_as_cs \
    --max-allowed-packet=67108864
