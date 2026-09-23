import pytest


@pytest.fixture(scope="session")
def spark():
    from src.data_prep import get_spark
    s = get_spark(app="cineinfer-tests", master="local[2]", driver_memory="1g",
                  shuffle_partitions=4)
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()
