# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Django 6.0.2 web application ("affinity") running on Python 3.14. Currently a fresh project scaffold with no custom apps. `djangorestframework` is installed but not yet added to `INSTALLED_APPS`.

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

# Create superuser
python manage.py createsuperuser
```

## Architecture

- **Django project package**: `affinity/` — settings, root URL conf, WSGI/ASGI entry points
- **Settings**: `affinity/settings.py` — single settings file, SQLite database, DEBUG=True
- **URL routing**: `affinity/urls.py` — root URL conf; only `/admin/` is configured
- **Database**: SQLite3 (`db.sqlite3`), no migrations applied yet

When adding new apps, remember to:
1. Add the app to `INSTALLED_APPS` in `affinity/settings.py`
2. Include the app's URLs in `affinity/urls.py` using `include()`
3. If using DRF, add `'rest_framework'` to `INSTALLED_APPS`
