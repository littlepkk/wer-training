import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _content_to_string(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def extract_system_content(trajectory: Any) -> str:
    system_contents: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("role") == "system":
                system_contents.append(_content_to_string(obj.get("content", "")))
                return

            for value in obj.values():
                walk(value)
            return

        if isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(trajectory)
    return "\n\n".join(system_contents)


def remove_system_messages(trajectory: Any) -> Any:
    def filter_obj(obj: Any) -> Any:
        if isinstance(obj, list):
            filtered_items = []
            for item in obj:
                if isinstance(item, dict) and item.get("role") == "system":
                    continue
                filtered_items.append(filter_obj(item))
            return filtered_items

        if isinstance(obj, dict):
            return {key: filter_obj(value) for key, value in obj.items()}

        return obj

    return filter_obj(deepcopy(trajectory))


def trajectory_without_system_to_string(trajectory: Any) -> str:
    filtered_trajectory = remove_system_messages(trajectory)
    return json.dumps(filtered_trajectory, ensure_ascii=False)


def write_extracted_parts(
    example_json_path: Path,
    system_content: str,
    trajectory_without_system: Any,
) -> tuple[Path, Path]:
    output_dir = example_json_path.parent
    system_output_path = output_dir / "system_content.json"
    trajectory_output_path = output_dir / "trajectory_without_system.json"

    system_output_path.write_text(
        json.dumps({"system_content": system_content}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trajectory_output_path.write_text(
        json.dumps(
            {"trajectory": trajectory_without_system},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return system_output_path, trajectory_output_path


def load_trajectory_from_example_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads("{" + raw + "}")

    if isinstance(data, dict) and "trajectory" in data:
        return data["trajectory"]
    return data
