docker build -t mymodel:latest .
docker run -d --name mymodel -p 80:80 mymodel:latest
