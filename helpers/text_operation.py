import textwrap


def wrap_dict_value_text(dict_in: dict, wrap_length: int = 25) -> dict:
    return {key: textwrap.fill(value, wrap_length) for key, value in dict_in.items()}
