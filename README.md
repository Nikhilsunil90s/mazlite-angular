# to install

Prerequisites
docker
git bash (if you have windows)


on windows
use powershell

to pull repo files (make sure you have added your ssh key to github account to access private repos otherwise will show no access)

in windows run 
.\docker\scripts\git-clone.bat

in mac/linux run 
sh ./docker/scripts/git-clone.bat



if you get duplicate key error when trying to add your ssh key to github account then you need to generate a new key for this

mac/linux
ssh-keygen -f mazlite-pem.key -t rsa -b 4096

then add mazlite-pem.key.pub to git account then run
export GIT_SSH_COMMAND='ssh -i ./mazlite-pem.key'
this will use above key when runnging git clone script above. To unset this variable do "unset GIT_SSH_COMMAND"
This instructions are for linux. for windows you need to install git bash


to initialize docker containers then run the following (This is only suppose to be run on initialization)

in windows:
.\docker\scripts\init-docker.bat


in mac/linux:
use terminal
sh docker/scripts/init-docker.sh


#to start docker run
docker-compose up -d


#to stop docker run
docker-compose down


#to run git pull
in windows
.\docker\scripts\git-pull.bat

in mac/linux
sh docker/scripts/git-pull.sh


#containers

web container
http://localhost:9220

to terminal into instance
docker exec -it mazlite-web-app /bin/bash

to force rebulid container and restart
docker-compose build --no-cache mazlite-web-app
docker-compose down
docker-compose up -d


api container
http://localhost:9221

to terminal into instance
docker exec -it mazlite-api-app /bin/bash

to force rebulid container and restart
docker-compose build --no-cache mazlite-api-app
docker-compose down
docker-compose up -d


mongo express
http://localhost:9223

to use mongo express use db credentials from docker-compose.yml
root
234i343k3

this is only for dev testing so not used for production


db:
to terminal into mysql
docker exec -it mazlite-db /bin/bash


DB MIGRATIONS:
to run migration do
docker exec -it mazlite-api-app sequelize-cli db:migrate

this must be done in powershell if running in windows



Troubleshooting
If for some reason containers are not running or showing node errors not finding libraries then reset all using

run:
docker/scripts/reset-all-docker.bat


if getting max depth exceeded then run
docker rmi -f $(docker images -a -q)

