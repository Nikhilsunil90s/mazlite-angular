cp ./docker/web/environment.ts ./mazlite-angular/src/environments/environment.ts
cp ./docker/api/docker-dev.sh ./mazlite-fastapi/docker-dev.sh
cp ./docker/api/docker-dev.sh ./mazlite-fastapi/run.sh
cp ./docker/api/Dockerfile ./mazlite-fastapi/Dockerfile

docker-compose build mazlite-web-app
docker-compose up -d

docker-compose build mazlite-fastapi-app
docker-compose up -d

#docker exec -i mazlite-fastapi-app apt install libgl1-mesa-glx

#docker exec -i mazlite-db sh /root/sample-db/import.sh


exit


#docker exec -it mazlite-fastapi-app /bin/bash
