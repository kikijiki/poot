"""A dependency-free validator for the Draft 7 keywords result.schema.json uses."""

import math
import re


def schema_errors(value, schema, path="$", root_schema=None):
    """Validate the Draft 7 keywords used by result.schema.json without a runtime dependency."""
    root_schema = root_schema or schema
    if "$ref" in schema:
        referenced = root_schema
        for component in schema["$ref"].removeprefix("#/").split("/"):
            referenced = referenced[component]
        schema = {
            **referenced,
            **{key: item for key, item in schema.items() if key != "$ref"},
        }
    errors = []
    type_matches = {
        "array": lambda item: isinstance(item, list),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
        ),
        "object": lambda item: isinstance(item, dict),
        "null": lambda item: item is None,
        "string": lambda item: isinstance(item, str),
    }
    expected_types = schema.get("type", [])
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if expected_types and not any(type_matches[name](value) for name in expected_types):
        return [f"{path} is not a schema {' or '.join(expected_types)}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} is not in the schema enum")
    if "pattern" in schema and isinstance(value, str) and not re.fullmatch(schema["pattern"], value):
        errors.append(f"{path} does not match the schema pattern")
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path} is below the schema minimum")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        errors.append(f"{path} is not above the schema exclusive minimum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path} has too few items")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(
                    schema_errors(
                        item, schema["items"], f"{path}[{index}]", root_schema
                    )
                )
    if isinstance(value, dict):
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"{path} is missing {field!r}")
        for field, field_schema in schema.get("properties", {}).items():
            if field in value:
                errors.extend(
                    schema_errors(
                        value[field], field_schema, f"{path}.{field}", root_schema
                    )
                )
    for subschema in schema.get("allOf", []):
        condition = subschema.get("if")
        if condition is None or not schema_errors(value, condition, path, root_schema):
            errors.extend(
                schema_errors(value, subschema.get("then", {}), path, root_schema)
            )
    return errors
