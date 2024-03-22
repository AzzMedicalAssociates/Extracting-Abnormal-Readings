FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Log the list of files in the working directory
RUN ls -l /code > /code/file_list.log

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
