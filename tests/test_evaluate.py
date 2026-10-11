from steam_reviewer.evaluate import _split_themes


def test_theme_list_saved_as_string():
    result = _split_themes("['gameplay', 'multiplayer']")

    assert result == {'gameplay', 'multiplayer'}


def test_semicolon_separated_themes():
    result = _split_themes("gameplay;multiplayer")

    assert result == {'gameplay', 'multiplayer'}


def test_empty_themes():
    assert _split_themes("") == set()
    assert _split_themes("[]") == set()


def test_invalid_themes():
    result = _split_themes("gameplay;invalid_theme")

    assert result == {'gameplay'}
