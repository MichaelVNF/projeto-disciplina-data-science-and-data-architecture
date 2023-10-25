def convert_to_boolean(value: str):
    if value == "0" or value.lower().startswith("no"):
        return False
    else:
        return True


def convert_to_float(value: str):
    if value.strip() == "":
        return 0.0
    else:
        return float(value)
