# mongo_io.py

# Imports
from pymongo import MongoClient
from pymongo.collection import Collection
from datetime import datetime
from typing import Optional, Dict


def get_collection(
                   uri: str = "mongodb://etl-mongo:27017",
                   db_name: str = "etl_metadata",
                   collection_name: str = "datasets"
                   ) -> Collection:
    """
    Returns a MongoDB collection used to store dataset metadata.

    Parameters
    ----------
    uri : str
        MongoDB connection URI.
    db_name : str
        Name of the MongoDB database.
    collection_name : str
        Name of the collection containing dataset metadata.

    Returns
    -------
    pymongo.collection.Collection
        MongoDB collection object.
    """
    client = MongoClient(uri)
    db = client[db_name]

    return db[collection_name]


def get_dataset_meta(
                     collection: Collection,
                     dataset_name: str
                     ) -> Optional[Dict]:
    """
    Retrieves metadata for a given dataset from MongoDB.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        MongoDB collection containing dataset metadata.
    dataset_name : str
        Logical name of the dataset.

    Returns
    -------
    dict or None
        Dataset metadata document if it exists, otherwise None.
    """

    return collection.find_one({"dataset": dataset_name})


def needs_update(
                 meta: Optional[Dict],
                 new_end_date: str
                 ) -> bool:
    """
    Determines whether a dataset needs to be updated based on metadata.

    Parameters
    ----------
    meta : dict or None
        Metadata document retrieved from MongoDB. If None, the dataset
        is assumed to be missing and requires processing.
    new_end_date : str
        Expected final date (YYYY-MM-DD) after update.

    Returns
    -------
    bool
        True if the dataset needs to be updated, False otherwise.
    """
    if meta is None:
        return True

    return meta.get("end_date", "") < new_end_date


def upsert_dataset_meta(
                        collection: Collection,
                        dataset_name: str,
                        path: str,
                        start_date: str,
                        end_date: str,
                        freq: str,
                        status: str = "ok"
                        ) -> None:
    """
    Inserts or updates dataset metadata in MongoDB.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        MongoDB collection containing dataset metadata.
    dataset_name : str
        Logical name of the dataset.
    path : str
        Filesystem path to the Zarr dataset.
    start_date : str
        Start date of the dataset (YYYY-MM-DD).
    end_date : str
        End date of the dataset (YYYY-MM-DD).
    freq : str
        Temporal frequency of the dataset (e.g., 'hourly', 'daily').
    status : str, optional
        Processing status of the dataset (default: "ok").

    Returns
    -------
    None
        Metadata is written to MongoDB as a side effect.
    """
    collection.update_one(
                          {"dataset": dataset_name},
                          {
                           "$set": {
                                    "dataset": dataset_name,
                                    "path": path,
                                    "start_date": start_date,
                                    "end_date": end_date,
                                    "freq": freq,
                                    "last_update": datetime.utcnow().strftime("%Y-%m-%d"),
                                    "status": status,
                                    }
                          },
                          upsert=True,
                          )
