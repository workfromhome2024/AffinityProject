# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Django 6.0.2 web application ("affinity") running on Python 3.14. Serves a SmolVLA robotics vision-language-action model as a REST API using Django REST Framework.

Key dependencies: `djangorestframework`, `lerobot` (Hugging Face LeRobot), `torch`, `Pillow`, `datasets`.

## Commands

```bash
# Run development server
python manage.py runserver

# Apply migrations
python manage.py migrate

# Create new migrations after model changes
python manage.py makemigrations

# Run all tests
python manage.py test

# Run tests for a specific app
python manage.py test <app_name>

# Run a single test
python manage.py test <app_name>.tests.<TestClass>.<test_method>

# Create a new app
python manage.py startapp <app_name>
```

## Architecture

- **Django project package**: `affinity/` — settings, root URL conf, WSGI/ASGI entry points
- **Settings**: `affinity/settings.py` — single settings file, SQLite database, DEBUG=True
- **Database**: SQLite3 (`db.sqlite3`)

### smolvla app

The `smolvla` app wraps the SmolVLA robotics policy model behind a DRF API endpoint.

- **Model loading** (`smolvla/apps.py`): `SmolVLAPolicy` from `lerobot` is loaded once at startup in `AppConfig.ready()` onto CPU. Only loads in the reloader child process (`RUN_MAIN=true`) to avoid double-loading in dev.
- **Inference endpoint** (`smolvla/views.py`): `POST /smolvla/api/predict/` — accepts a multipart form with an `image` file and optional `instruction` text. Returns a JSON action chunk from the model.
- **Test data helper** (`smolvla/tests.py`): `download_and_sample_vla_data()` fetches sample data from `lerobot/svla_so100_stacking` on Hugging Face for testing.

**Note**: `smolvla` and `rest_framework` are not yet added to `INSTALLED_APPS` in settings.py — they need to be added for the app to function.

### URL routing

- `/admin/` — Django admin
- `/smolvla/` — includes `smolvla.urls` (configured in root `affinity/urls.py`)
