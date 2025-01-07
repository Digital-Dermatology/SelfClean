import toml


def parse_requirements(file_path):
    """Parse requirements.txt into a dictionary."""
    dependencies = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                dependencies.append(line)
    return dependencies


def update_pyproject_toml(pyproject_path, dependencies):
    """Update pyproject.toml with dependencies in the [project] section."""
    data = toml.load(pyproject_path)

    # Ensure the `[project]` section exists
    if "project" not in data:
        data["project"] = {}

    # Ensure the `dependencies` key exists
    if "dependencies" not in data["project"]:
        data["project"]["dependencies"] = []

    # Add dependencies to the `dependencies` key
    # Avoid duplicates by converting to a set
    existing_deps = set(data["project"]["dependencies"])
    new_deps = set(dependencies)
    updated_deps = sorted(existing_deps.union(new_deps))  # Sorted for consistency
    data["project"]["dependencies"] = updated_deps

    # Write back to the pyproject.toml file
    with open(pyproject_path, "w") as f:
        toml.dump(data, f)


def main():
    requirements_path = "requirements.txt"
    pyproject_path = "pyproject.toml"

    # Parse the requirements.txt
    dependencies = parse_requirements(requirements_path)

    # Update the pyproject.toml
    update_pyproject_toml(pyproject_path, dependencies)
    print(f"Updated {pyproject_path} with dependencies from {requirements_path}")


if __name__ == "__main__":
    main()
