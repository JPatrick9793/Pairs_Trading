# Purpose

This is a basic template for creating and manipulating SQLAlchemy databases and tables. This module
is configurable, so you can change some settings in the **config.ini** file, or even add your own if you wish
to extend this template.

# How to use

The main file is **db_config.py** and it serves two purposes:

1. If run like a script, it will create a SQLite database (location specified by config.ini)
2. If imported, provides utility functions for CRUD operations on said database.

# Setup

All packages required to use this module are included in the requirements.txt file.
