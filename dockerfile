# use python as base image
FROM python:3.11-slim

# copy over package requirements
COPY ["src/Configuration Files/requirements.txt", "./requirements.txt"]
RUN pip install -r requirements.txt
#Copy files to your container
COPY src /src

#Running your APP and doing some PORT Forwarding
EXPOSE 8050

WORKDIR "/src"

ENTRYPOINT ["python3"]
CMD ["app.py"]