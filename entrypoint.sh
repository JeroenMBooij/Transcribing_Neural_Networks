#!/bin/sh

python manage.py collectstatic --noinput

gunicorn src.wsgi:application --bind 0.0.0.0:8000