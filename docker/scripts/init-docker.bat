rem docker-compose build --no-cache mazlite-fastapi-app
copy .\docker\web\environment.ts .\mazlite-angular\src\environments\environment.ts
copy .\docker\api\docker-dev.sh .\mazlite-fastapi\docker-dev.sh
copy .\docker\api\docker-dev.sh .\mazlite-fastapi\run.sh
copy .\docker\api\Dockerfile .\mazlite-fastapi\Dockerfile

docker-compose build mazlite-web-app
docker-compose up -d

docker-compose build mazlite-fastapi-app
docker-compose up -d

rem docker exec -i mazlite-fastapi-app apt install libgl1-mesa-glx

rem docker exec -i mazlite-db sh /root/sample-db/import.sh


exit


rem docker exec -it mazlite-fastapi-app /bin/bash
