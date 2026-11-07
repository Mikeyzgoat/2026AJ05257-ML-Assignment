from src.models import MODEL_FILES, build_models


def test_five_required_models_exist():
    assert len(build_models()) == 5
    assert set(build_models()) == set(MODEL_FILES)

