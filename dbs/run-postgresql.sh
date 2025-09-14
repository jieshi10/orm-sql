docker run --rm -it -d \
    -e POSTGRES_PASSWORD=YOUR_PASSWORD \
    -v $(pwd)/bird_pg_db/:/var/lib/postgresql/data \
    -p 5432:5432 \
    postgres:17.2
