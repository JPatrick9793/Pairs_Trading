# -*- coding: utf-8 -*-
"""
This script serves two purposes:

    1. If run like a script, it will create a SQLite database (location specified by config.ini)
    2. If imported, provides utility functions for CRUD operations on said database.

"""
import configparser
from typing import List, Dict, Tuple, Callable, Any
from pathlib import Path

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm.query import Query

Base: DeclarativeMeta = declarative_base()

# Global variable settings
THIS_DIRECTORY: Path = Path(__file__).parent                #: Path object pointing to this file's directory.

# Parse the config file
cp = configparser.ConfigParser()                            #: config parser object.
cp.read(THIS_DIRECTORY / 'config.ini')

# Global Variables
DB_PATH: Path = THIS_DIRECTORY / cp['CONFIG']['db_path']    #: Path object for location of database.

ENGINE = create_engine('sqlite:///{}'.format(DB_PATH))      #: SQLAlchemy Connection Engine.
DB_SESSION = sessionmaker(bind=ENGINE)                      #: Class for creating new SQLAlchemy sessions.


class Test1(Base):
    """
    Test1 database. Simple Example.
    """

    __tablename__ = "Test1"

    key_int = Column(Integer, primary_key=True, autoincrement=True)     #: auto-incrementing Primary Key (Int)

    col_int = Column(Integer)                                           #: Integer Column
    col_float = Column(Float)                                           #: Float Column
    col_string = Column(String)                                         #: String Column


class Test2(Base):
    """
    Test2 Table. Simple example of using multiple primary keys together.
    """

    __tablename__ = "Test2"

    key_string1 = Column(String, primary_key=True)                      #: String Primary Key
    key_string2 = Column(String, primary_key=True)                      #: String Primary Key

    col_int = Column(Integer)                                           #: Integer Column
    col_float = Column(Float)                                           #: Float Column
    col_string = Column(String)                                         #: String Column


def fix_table_type(func: Callable):
    """
    Decorator for converting table function argument from string to Base Object.

    Wrapping functions such as query_table() with this allows for either a String or Base subclass
    objects to be passed in for 'table' parameter.

    :param func: Callable function to be wrapped
    :return:
    """
    def wrapper(*args: Tuple, **kwargs):
        # Convert string into actual Class object
        if isinstance(kwargs['table'], str):
            table_string: str = kwargs.pop('table')
            table = globals()[table_string]
            return func(*args, table=table, **kwargs)
        # table arg was already Base object
        else:
            return func(*args, **kwargs)
    return wrapper


def create_database(file_path: str):
    """
    Function to create a SQLite DataBase. The tables are based on the classes defined in this file.

    :param file_path: file path to the database. (Example: 'sqlite_example.db')
    :return:
    """
    path_to_db_dir: Path = Path(file_path).resolve().parent
    path_to_db_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(ENGINE)


@fix_table_type
def query_table(table: Base, filter: Dict = None, one: bool = False):
    """
    Query a table in the database using a filter. Returns one or many results depending on 'one'
    :param table: (String or Base-subclass) Name of table to be queried.
    :param filter: (bool, optional) condition to filter the results by.
    :param one: (bool, optional) whether to return one, or many results
    :return: Either a dictionary (one = true) or list of dictionaries (one = false)
    """

    # Create session
    session: Session = get_session()

    # Begin query steps
    query: Query = session.query(table)
    if filter is not None:
        query: Query = query.filter_by(**filter)

    if one:
        query: Base = query.one()
        results: Dict[str, Any] = _convert_row_to_dict(query)
    elif not one:
        query: Base = query.all()
        results: List[Dict[str, Any]] = [_convert_row_to_dict(q) for q in query]
    else:
        raise ValueError('Parameter \'one\' given invalid value.')

    # close sesion and return the query results
    session.close()
    return results


def get_session():
    return DB_SESSION()


@fix_table_type
def add_entry_single(table: Base, entry: Dict):
    """
    Creates new SQLAlchemy session and commits entry to database in table
    :return:
    """
    session = get_session()

    new_entry = table(**entry)
    session.add(new_entry)

    session.commit()
    session.close()


@fix_table_type
def add_entry_multiple(table: Base, entries: List[Dict]):
    """
    Creates new SQLAlchemy session and commits entry to database in table
    :return:
    """

    session = get_session()

    for entry in entries:
        new_entry = table(**entry)
        session.add(new_entry)

    session.commit()


def _convert_row_to_dict(row: Base) -> Dict[str, Any]:
    """
    Converts query results into python dictionary for easy manipulation
    :param row: Base subclass query result.
    :return:
    """
    return {c.name: str(getattr(row, c.name)) for c in row.__table__.columns}


def main():
    """
    Main script. This function gets called when the file is run as a script.
    """

    # Delete existing database
    DB_PATH.unlink()

    # create database
    create_database(file_path=str(DB_PATH))

    # Add test entries to first table
    first_table_entry1 = {'col_int': 1, 'col_float': 0.1, 'col_string': 'test_single'}
    add_entry_single(table=Test1, entry=first_table_entry1)
    add_entry_single(table="Test1", entry=first_table_entry1)

    # Add test entries to second table
    add_entry_single(
        table=Test2,
        entry={'key_string1': 'a',
               'key_string2': 'a',
               'col_int': 1,
               'col_float': 0.1,
               'col_string': 'test'})
    add_entry_single(
        table="Test2",
        entry={'key_string1': 'a',
               'key_string2': 'b',
               'col_int': 1,
               'col_float': 0.1,
               'col_string': 'test'})

    # Try adding multiple to
    first_table_entry2 = {'col_int': 3, 'col_float': 0.3, 'col_string': 'test_multiple'}
    add_entry_multiple(table='Test1', entries=[first_table_entry2 for _ in range(5)])

    # test table query single
    entry: Dict = query_table(table="Test1", filter={'key_int': 1}, one=True)
    print('\n', entry)

    # test table query multiple
    entry: Dict = query_table(table="Test1", filter={'col_int': 1}, one=False)
    print('\n', entry)


if __name__ == "__main__":

    main()
