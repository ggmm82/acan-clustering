"""Gunicorn entry point; the single worker owns all training sessions."""
from .render import app

application = app.server
