#JASS URLS
http://jass-server.abiz.ch/games/played
https://www.swisslos.ch/de/jass/informationen/jass-regeln/jass-grundlagen.html
https://jass1.schieber.ch/


#List of commands

### build image
docker-compose build

### run container
docker-compose up -d

### inspect container
docker-compose logs -f


## Heroku
https://id.heroku.com/login

https://deep-jass-project-generations.herokuapp.com/
For the Player url add /player_name from the string in the app.py

### push image to heroku
heroku login
heroku container:login

### push App
heroku container:push web -a deep-jass-project-generations

### release App
heroku container:release web -a deep-jass-project-generations

### attach to log
heroku logs --tail -a deep-jass-project-generations

### fuel/money to run 
heroku ps:scale web=1 -a deep-jass-project-generations