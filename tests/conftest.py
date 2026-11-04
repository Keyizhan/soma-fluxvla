import pytest


@pytest.fixture(autouse=True)
def _subtask_config_roots(monkeypatch, tmp_path_factory):
    """Example subtask configs use ${SOMA_SUBTASK_ROOT}/${SOMA_PI05_BASE} (D19); give tests a
    placeholder root so loading them works. Real runs must set the variables themselves."""
    import os
    root = tmp_path_factory.getbasetemp() / 'subtask_root'
    for name, value in (('SOMA_SUBTASK_ROOT', root), ('SOMA_PI05_BASE', root / 'pi05_base')):
        if not os.environ.get(name):
            monkeypatch.setenv(name, str(value))
