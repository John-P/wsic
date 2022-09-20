from pathlib import Path

import pytest


@pytest.fixture()
def samples_path():
    """Return the path to the samples."""
    return Path(__file__).parent / "samples"


def pytest_generate_tests(metafunc):
    """Generate test scenarios.

    See
    https://docs.pytest.org/en/7.1.x/example/parametrize.html#a-quick-port-of-testscenarios
    """
    id_list = []
    arg_values = []
    if metafunc.cls is None:
        return
    for scenario in metafunc.cls.scenarios:
        id_list.append(scenario[0])
        items = scenario[1].items()
        arg_names = [x[0] for x in items]
        arg_values.append([x[1] for x in items])
    metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="class")
