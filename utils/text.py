from typing import Any


def dict_to_text(data: dict, indent_level: int = 0) -> str:
    """Recursively converts a dictionary to a clear, readable text format."""
    lines = []
    indent = "  " * indent_level
    for key, value in data.items():
        if value is None or value == "":
            continue

        clean_key = str(key).replace("_", " ").title()

        if isinstance(value, dict):
            lines.append(f"{indent}{clean_key}:")
            lines.append(dict_to_text(value, indent_level + 1))
        elif isinstance(value, list):
            if not value:
                continue
            lines.append(f"{indent}{clean_key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{indent}  -")
                    nested_text = dict_to_text(item, indent_level + 2)
                    lines.append(nested_text)
                else:
                    lines.append(f"{indent}  - {item}")
        else:
            lines.append(f"{indent}{clean_key}: {value}")

    return "\n".join(lines)


def convert_entity_to_text(entity: Any) -> str:
    """
    Converts any Pydantic model or dictionary to a text representation
    suitable for RAG context ingestion.
    """
    if hasattr(entity, "model_dump"):
        data = entity.model_dump()
    elif hasattr(entity, "dict"):
        data = entity.dict()
    elif isinstance(entity, dict):
        data = entity
    else:
        return str(entity)

    return dict_to_text(data)
