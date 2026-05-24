import toml
from packaging import version
import re

def get_evaluation_values():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    table = re.search(r"(?<=## Evaluation\n\n)[\s\S]*", content).group(0)
    pattern = r"\|(\s*[\w\s]+)\s*\|\s*\*\*(\d+\.\d+)\%\*\*\s*\|"
    matches = re.findall(pattern, table)
    evaluation_values = {module.strip(): float(value) for module, value in matches}
    return evaluation_values

def define_env(env):
    env.variables.pretrained_models = "https://huggingface.co/roshan-research/models"
    
    evals = get_evaluation_values()
    env.variables.lemmatizer_evaluation_value = evals.get("Lemmatizer", 0)
    env.variables.posTagger_evaluation_value = evals.get("POSTagger", 0)
    env.variables.dependency_parser_evaluation_value = evals.get("DependencyParser", 0)
    env.variables.chunker_evaluation_value = evals.get("Chunker", 0)

    @env.macro
    def needed_python_version():
        """Automatically gets the minimum python version from pyproject.toml"""
        with open('pyproject.toml', 'r') as f:
            toml_data = toml.load(f)
        
        python_reqs = toml_data.get('tool', {}).get('poetry', {}).get('dependencies', {}).get('python', '')
        versions = [version.parse(c.split('>=')[1]) for c in python_reqs.split(',') if c.startswith('>=')]
        min_version = str(min(versions)) if versions else '3.12'
        return f"{min_version}+"

    @env.macro
    def hazm_code_example():
        """Automatically gets the usage example code from README.md"""
        with open("README.md", 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        
        pattern = r"(?<=## Usage\n\n)```python[\s\S]*```"    
        match = re.search(pattern, markdown_content)
        return match.group(0) if match else ""