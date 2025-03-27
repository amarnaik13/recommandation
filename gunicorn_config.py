# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 3
timeout = 3600  # Set the timeout to 120 seconds
