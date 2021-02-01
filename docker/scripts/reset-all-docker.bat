docker-compose down &
docker-compose build --no-cache mazlite-fastapi-app
docker-compose build --no-cache mazlite-web-app
docker-compose up -d
