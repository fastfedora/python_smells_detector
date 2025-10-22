import os
import ast
import networkx as nx
import re
from collections import defaultdict
import yaml
from dataclasses import dataclass
import sys
import importlib.util
import logging
from .exceptions import CodeAnalysisError

# Set up logger
logger = logging.getLogger(__name__)

@dataclass
class ArchitecturalSmell:
    name: str
    description: str
    file_path: str
    module_class: str
    line_number: int = None
    severity: str = 'medium'

class ArchitecturalSmellDetector:
    """
    A class to detect architectural smells in Python projects.

    This class analyzes Python source code files within a directory to identify
    various architectural smells based on predefined thresholds.

    Attributes:
        architectural_smells (list): A list to store detected architectural smells.
        module_dependencies (nx.DiGraph): A directed graph to represent module dependencies.
        module_functions (defaultdict): A dictionary to store functions for each module.
        api_usage (defaultdict): A dictionary to store API usage for each module.
        thresholds (dict): A dictionary of threshold values for various smell detections.
        file_paths (dict): A dictionary to store file paths for each module.
        external_dependencies (dict): A dictionary to store external dependencies for each module.
        function_calls (defaultdict): A dictionary to track inter-module function calls.
    """

    def __init__(self, thresholds):
        """
        Initialize the ArchitecturalSmellDetector with given thresholds.

        Args:
            thresholds (dict): A dictionary of threshold values for various smell detections.
        """
        self.architectural_smells = []
        self.module_dependencies = nx.DiGraph()
        self.module_functions = defaultdict(set)
        self.api_usage = defaultdict(list)
        self.thresholds = thresholds
        self.file_paths = {}  # New attribute to store file paths
        self.project_modules = set()
        self.entry_point_modules = set()
        self.external_dependencies = defaultdict(set)
        self.function_calls = defaultdict(set)  # Track inter-module function calls

    def load_thresholds(self, config_path):
        """
        Load threshold values from a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: A dictionary of threshold values for architectural smells.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return {k: v['value'] for k, v in config['architectural_smells'].items()}

    def detect_smells(self, directory_path):
        """
        Detect architectural smells in the given directory.
        """
        detection_methods = [
            (self.detect_hub_like_dependency, "detect_hub_like_dependency"),
            (self.detect_scattered_functionality, "detect_scattered_functionality"),
            (self.detect_redundant_abstractions, "detect_redundant_abstractions"),
            (self.detect_god_objects, "detect_god_objects"),
            (self.detect_improper_api_usage, "detect_improper_api_usage"),
            (self.detect_orphan_modules, "detect_orphan_modules"),
            (self.detect_cyclic_dependencies, "detect_cyclic_dependencies"),
            (self.detect_unstable_dependencies, "detect_unstable_dependencies")
        ]

        try:
            # First analyze the directory structure
            logger.info(f"Analyzing directory structure: {directory_path}")
            self.analyze_directory(directory_path)

            # Then run each detection method
            for detect_method, method_name in detection_methods:
                try:
                    logger.debug(f"Running {method_name}")
                    detect_method()
                except Exception as e:
                    logger.error(f"Error in {method_name}: {str(e)}", exc_info=True)
                    raise CodeAnalysisError(
                        message=str(e),
                        file_path=directory_path,
                        function_name=method_name
                    )

        except Exception as e:
            logger.error(f"Error analyzing directory {directory_path}: {str(e)}", exc_info=True)
            raise CodeAnalysisError(
                message=str(e),
                file_path=directory_path
            )

    def analyze_directory(self, directory_path):
        """
        Analyze all Python files in the given directory and its subdirectories.

        Args:
            directory_path (str): The path to the directory to be analyzed.
        """
        # Find the actual project root for consistent module naming
        project_root = self._find_project_root(directory_path)

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path, project_root=project_root)

        # After analyzing all files, resolve external dependencies
        self.resolve_external_dependencies(project_root=project_root)

    def analyze_file(self, file_path, project_root=None):
        """
        Analyze a single Python file for architectural information with improved
        intra-project dependency detection.
        """
        try:
            with open(file_path, 'r') as file:
                tree = ast.parse(file.read())

            # Get relative module path using the project root for consistency
            if not project_root:
                project_root = self._find_project_root(file_path)
            module_name = os.path.relpath(file_path, project_root)
            module_name = module_name.replace(os.path.sep, '.')[:-3]  # Remove .py extension
            self.module_dependencies.add_node(module_name)
            self.project_modules.add(module_name)
            self.file_paths[module_name] = file_path
            if self.is_entry_point(tree):
                self.entry_point_modules.add(module_name)

            # Track local imports and their line numbers
            local_imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        local_imports.append((import_name, node.lineno))
                        self.module_dependencies.add_edge(module_name, import_name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:  # This is a relative import
                            current_package = module_name.split('.')
                            # Go up by node.level
                            parent_package = '.'.join(current_package[:-node.level])
                            if parent_package:
                                import_name = f"{parent_package}.{node.module}"
                            else:
                                import_name = node.module
                        else:
                            import_name = node.module

                        local_imports.append((import_name, node.lineno))
                        self.module_dependencies.add_edge(module_name, import_name)
                        # Track imported names for more detailed dependency analysis
                        for alias in node.names:
                            if alias.name != '*':
                                full_import = f"{import_name}.{alias.name}"
                                self.module_functions[import_name].add(alias.name)


                elif isinstance(node, ast.FunctionDef):
                    self.module_functions[module_name].add(node.name)

                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        name, context, context_type = self.resolve_api_call(node.func, local_imports)
                        self.api_usage[module_name].append((name, context, context_type))

                        # Track function calls between modules
                        if isinstance(node.func.value, ast.Name):
                            # Check if this is a call to an imported module
                            module_called = node.func.value.id
                            if any(module_called == imp[0].split('.')[-1] for imp in local_imports):
                                self.function_calls[module_name].add((module_called, node.func.attr))

        except SyntaxError as e:
            print(f"Parse error in file {file_path}: {str(e)}")
        except Exception as e:
            print(f"Error analyzing file {file_path}: {str(e)}")

    def resolve_external_dependencies(self, project_root=None):
        """
        Resolve external dependencies while preserving intra-project dependencies.
        """
        # Get all project modules
        if not project_root:
            project_root = self._find_project_root(next(iter(self.file_paths.values())))
        standard_lib_modules = set(sys.stdlib_module_names)

        for module in list(self.module_dependencies.nodes()):
            if module not in self.project_modules:
                is_stdlib = any(
                    module == std_lib or module.startswith(f"{std_lib}.")
                    for std_lib in standard_lib_modules
                )
                if is_stdlib:
                    module_type = 'stdlib'
                else:
                    try:
                        spec = importlib.util.find_spec(module.split('.')[0])
                        if spec is not None:
                            module_type = 'third_party'
                    except (ModuleNotFoundError, ValueError):
                        module_type = 'unknown'

                # Convert module to external dependency
                self.module_dependencies.remove_node(module)
                self.external_dependencies[module].add((module_type, module))

    def resolve_api_call(self, func_node, local_imports):
        """
        Resolve API calls to their name, context, and context type based on AST analysis.

        Args:
            func_node (ast.Name or ast.Attribute): The AST node representing the function call
            local_imports (list): List of (import_name, line_number) tuples for imported modules

        Returns:
            tuple: (name, context, context_type) where:
                - name (str): The function/method name (last element of the path)
                - context (str): The full API call path for context
                - context_type (str): The context type indicating how the call was resolved

        Raises:
            ValueError: If the function node type is not a Name or Attribute.

        Context Types and Examples:
            - "none": Direct function calls with no object context
                print("hello") -> ("print", None, "none")
                len(my_list) -> ("len", None, "none")

            - "imported-symbol": Calls to imported modules/classes
                sqlite3.connect("db") -> ("connect", "sqlite3", "imported-symbol")
                requests.get("url") -> ("get", "requests", "imported-symbol")

            - "local-symbol": Calls to local variables or other symbol
                cursor.execute("sql") -> ("execute", "cursor", "local-symbol")
                my_list.append("item") -> ("append", "my_list", "local-symbol")

            - "attribute-chain": Chained attribute access
                obj.subobj.method() -> ("method", "obj.subobj", "attribute-chain")
                request.headers.get() -> ("get", "request.headers", "attribute-chain")

            - "call-chain": Method calls on function return values
                hashlib.sha256().hexdigest() -> ("hexdigest", "hashlib.sha256()", "call-chain")
                datetime.now().strftime("%Y") -> ("strftime", "datetime.now()", "call-chain")

            - "subscript": Method calls on subscripted objects
                results['errors'].append() -> ("append", "results['errors']", "subscript")
                data['users'].extend() -> ("extend", "data['users']", "subscript")

            - "expression": Method calls on complex expressions
                (password + salt).encode() -> ("encode", "<BinOp>", "expression")
                (obj1 + obj2).method() -> ("method", "<BinOp>", "expression")
                ", ".join(my_list) -> ("join", "<Constant>", "expression")
        """
        if isinstance(func_node, ast.Name):
            # Direct function call, e.g., print("hello")
            return (func_node.id, None, "none")

        elif isinstance(func_node, ast.Attribute):
            method_name = func_node.attr

            if isinstance(func_node.value, ast.Name):
                symbol = func_node.value.id

                # Check if this is a call to from imported module/class
                import_info = self._find_import_for_symbol(symbol, local_imports)
                if import_info:
                    # Call from imported module/class, e.g., sqlite3.connect("db")
                    return (method_name, import_info, "imported-symbol")
                else:
                    # Call from local variable or other symbol, e.g., cursor.execute("sql")
                    return (method_name, symbol, "local-symbol")

            elif isinstance(func_node.value, ast.Attribute):
                # Call from attribute chain, e.g., obj.subobj.method
                base_path = self._resolve_ast_node(func_node.value)
                return (method_name, base_path, "attribute-chain")

            elif isinstance(func_node.value, ast.Call):
                # Call from function call chain, e.g., hashlib.sha256().hexdigest
                func_call = self._resolve_ast_node(func_node.value)
                return (method_name, func_call, "call-chain")

            elif isinstance(func_node.value, ast.Subscript):
                # Call from subscripted object, e.g., results['errors'].append
                subscript_path = self._resolve_ast_node(func_node.value)
                return (method_name, subscript_path, "subscript")

            else:
                # Call from complex expression, e.g., (password + salt).encode()
                # Since expressions may not group properly, we use the AST node type name instead
                node_type = type(func_node.value).__name__
                return (method_name, f"<{node_type}>", "expression")

        else:
            raise ValueError(f"Unsupported function node type: {type(func_node)}")

    def add_smell(self, name, description, file_path, module_class, line_number=None, severity='medium'):
        """
        Add a detected architectural smell to the list.

        Args:
            name (str): The name of the smell
            description (str): Description of the smell
            file_path (str): Path to the file containing the smell
            module_class (str): The module or class containing the smell
            line_number (int, optional): The line number where the smell was detected
            severity (str, optional): The severity level of the smell (default: 'medium')
        """
        self.architectural_smells.append(ArchitecturalSmell(
            name=name,
            description=description,
            file_path=file_path,
            module_class=module_class,
            line_number=line_number,
            severity=severity
        ))

    def detect_hub_like_dependency(self):
        """
        Detect hub-like dependencies in the project with improved accuracy.
        """
        total_modules = len(self.module_dependencies.nodes())
        if total_modules < 3:  # Skip analysis for very small projects
            return

        threshold = self.thresholds.get('HUB_LIKE_DEPENDENCY_THRESHOLD', 0.5)
        min_connections = self.thresholds.get('MIN_HUB_CONNECTIONS', 5)

        for node in self.module_dependencies.nodes():
            # Exclude common infrastructure modules
            if (
                any(pattern in node.lower() for pattern in ['util', 'common', 'base', 'core']) or
                node in self.entry_point_modules # Entry points should be hub-like
            ):
                continue

            # Count both internal and external dependencies
            in_degree = self.module_dependencies.in_degree(node)
            out_degree = self.module_dependencies.out_degree(node)
            external_deps = len(self.external_dependencies[node])
            total_connections = in_degree + out_degree + external_deps

            # Calculate fan-in and fan-out ratios
            fan_in_ratio = in_degree / total_modules if total_modules > 0 else 0
            fan_out_ratio = (out_degree + external_deps) / total_modules if total_modules > 0 else 0

            # Check for hub-like characteristics
            is_hub = (total_connections >= min_connections and
                     (total_connections / total_modules) > threshold)

            # Additional checks to reduce false positives
            if is_hub:
                # Check if the module has balanced dependencies
                is_balanced = 0.2 <= fan_in_ratio / (fan_out_ratio + 0.0001) <= 5

                if not is_balanced:
                    self.add_smell(
                        "Hub-like Dependency",
                        f"Module '{node}' is a potential hub with {total_connections} connections "
                        f"(in: {in_degree}, out: {out_degree}, external: {external_deps})",
                        self.file_paths.get(node, "Unknown"),
                        node,
                        severity='high' if total_connections > min_connections * 2 else 'medium'
                    )

    def detect_scattered_functionality(self):
        """
        Detect scattered functionality in the project.
        """
        function_modules = defaultdict(list)
        min_function_length = 3  # Ignore very short function names
        excluded_names = {'main', 'init', 'setup', 'test'}  # Common function names to exclude
        for module, functions in self.module_functions.items():

            for func in functions:
                # Skip common/utility functions and short names
                if (len(func) >= min_function_length and
                    func.lower() not in excluded_names and
                    not func.startswith('_')):  # Skip private functions
                    function_modules[func].append(module)

        min_occurrences = self.thresholds.get('MIN_SCATTERED_OCCURRENCES', 3)
        for func, modules in function_modules.items():
            if len(modules) >= min_occurrences:  # Increase minimum occurrences threshold
                self.add_smell(
                    "Scattered Functionality",
                    f"Function '{func}' appears in {len(modules)} modules: {', '.join(modules)}",
                    self.file_paths.get(modules[0], "Unknown"),
                    modules[0]
                )

    def detect_redundant_abstractions(self):
        """
        Detect potential redundant abstractions in the project.
        """
        similar_modules = defaultdict(list)
        min_functions = 3  # Minimum number of functions to consider

        for module, functions in self.module_functions.items():
            # Only consider modules with sufficient functions
            if len(functions) >= min_functions:
                # Filter out private functions and common utility functions
                public_functions = {f for f in functions
                                 if not f.startswith('_')
                                 and len(f) > 3
                                 and f.lower() not in {'main', 'init', 'setup', 'test'}}

                if public_functions:  # Only proceed if there are public functions
                    signature = frozenset(public_functions)
                    similar_modules[signature].append(module)

        similarity_threshold = self.thresholds.get('REDUNDANT_SIMILARITY_THRESHOLD', 0.8)
        for signature, modules in similar_modules.items():
            if len(modules) > 1 and len(signature) >= min_functions:
                # Calculate similarity score between modules
                for i in range(len(modules)):
                    for j in range(i + 1, len(modules)):
                        module1_funcs = self.module_functions[modules[i]]
                        module2_funcs = self.module_functions[modules[j]]
                        similarity = len(module1_funcs & module2_funcs) / len(module1_funcs | module2_funcs)

                        if similarity >= similarity_threshold:
                            self.add_smell(
                                "Potential Redundant Abstractions",
                                f"Modules {modules[i]} and {modules[j]} have {similarity:.1%} similar functionalities",
                                self.file_paths.get(modules[i], "Unknown"),
                                modules[i]
                            )

    def detect_god_objects(self):
        """
        Detect god objects in the project.
        """
        min_functions = self.thresholds.get('MIN_GOD_OBJECT_FUNCTIONS', 5)
        excluded_patterns = {'test_', 'setup_', 'config_'}  # Common prefixes to exclude

        for module, functions in self.module_functions.items():
            # Filter out private methods and common test/setup functions
            public_functions = {f for f in functions
                              if not f.startswith('_') and
                              not any(f.startswith(pattern) for pattern in excluded_patterns)}

            if (len(public_functions) >= min_functions and
                len(public_functions) > self.thresholds['GOD_OBJECT_FUNCTIONS']):
                self.add_smell(
                    "God Object",
                    f"Module '{module}' has too many public functions ({len(public_functions)})",
                    self.file_paths.get(module, "Unknown"),
                    module
                )

    def detect_improper_api_usage(self):
        """
        Detect potential improper API usage in the project.
        """
        min_calls = self.thresholds.get('MIN_API_CALLS', 10)  # Minimum calls to consider
        repetition_threshold = self.thresholds.get('API_REPETITION_THRESHOLD', 0.4)
        use_full_api_call_path = self.thresholds.get('USE_FULL_API_CALL_PATH', False)
        exclude_api_call_paths = self.thresholds.get('EXCLUDE_API_CALL_PATHS', [])

        for module, api_calls in self.api_usage.items():
            if len(api_calls) >= min_calls:
                # Count frequency of each API call
                call_frequency = {}
                for call, context, context_type in api_calls:
                    call_path = f"{context}.{call}" if use_full_api_call_path and context else call
                    if not any(re.search(pattern, call_path) for pattern in exclude_api_call_paths):
                        call_frequency[call_path] = call_frequency.get(call_path, 0) + 1

                # Check for highly repetitive calls
                repetitive_calls = {call: count for call, count in call_frequency.items()
                                  if count >= 3}  # Ignore calls repeated less than 3 times

                repetitive_calls_ratio = sum(repetitive_calls.values()) / len(api_calls)
                if repetitive_calls_ratio > repetition_threshold:
                    self.add_smell(
                        "Potential Improper API Usage",
                        f"Module '{module}' has {repetitive_calls_ratio:.1%} repetitive API calls: " +
                        ", ".join(f"{call}({count}x)" for call, count in repetitive_calls.items()),
                        self.file_paths.get(module, "Unknown"),
                        module
                    )

    def detect_orphan_modules(self):
        """
        Detect orphan modules in the project.
        """
        excluded_modules = {'__init__', 'setup', 'tests', 'utils'}  # Common standalone modules
        min_project_size = self.thresholds.get('MIN_PROJECT_SIZE', 3)

        if len(self.module_dependencies.nodes()) < min_project_size:
            return

        for node in self.module_dependencies.nodes():
            module_name = node.split('.')[-1]
            # Fix: Check if any excluded module name is in the full node path
            if (self.module_dependencies.in_degree(node) + self.module_dependencies.out_degree(node) == 0 and
                module_name not in excluded_modules and
                not any(excluded in node.lower() for excluded in excluded_modules)):
                self.add_smell(
                    name="Orphan Module",
                    description=f"'{node}' is isolated from other modules",
                    file_path=self.file_paths.get(node, "Unknown"),
                    module_class=node,
                    severity='medium'
                )

    def detect_cyclic_dependencies(self):
        """
        Detect cyclic dependencies with improved accuracy and cycle classification.
        """
        min_cycle_size = self.thresholds.get('MIN_CYCLE_SIZE', 2)
        max_cycle_size = self.thresholds.get('MAX_CYCLE_SIZE', 5)
        excluded_modules = {'__init__', 'utils', 'common', 'base', 'core'}

        # Find all simple cycles
        cycles = list(nx.simple_cycles(self.module_dependencies))

        # Group cycles by their shared nodes to identify related cycles
        cycle_groups = defaultdict(list)

        for cycle in cycles:
            if min_cycle_size <= len(cycle) <= max_cycle_size:
                # Skip cycles containing excluded modules
                if any(any(excluded in node.lower() for excluded in excluded_modules)
                      for node in cycle):
                    continue

                # Calculate cycle metrics
                cycle_strength = 0
                for i in range(len(cycle)):
                    node1 = cycle[i]
                    node2 = cycle[(i + 1) % len(cycle)]
                    # Count mutual dependencies
                    cycle_strength += sum(1 for _ in nx.all_simple_paths(
                        self.module_dependencies, node1, node2))

                # Group related cycles
                cycle_key = frozenset(cycle)
                cycle_groups[cycle_key].append((cycle, cycle_strength))

        # Report cycles with additional context
        for cycle_group in cycle_groups.values():
            strongest_cycle = max(cycle_group, key=lambda x: x[1])
            cycle, strength = strongest_cycle

            # Calculate severity based on cycle size and strength
            severity = 'high' if len(cycle) >= 3 and strength >= 3 else 'medium'

            cycle_str = ' -> '.join(cycle + [cycle[0]])
            self.add_smell(
                "Cyclic Dependency",
                f"Strong cyclic dependency detected: {cycle_str}\n"
                f"Cycle strength: {strength} mutual dependencies",
                self.file_paths.get(cycle[0], "Unknown"),
                cycle[0],
                severity=severity
            )

    def detect_unstable_dependencies(self):
        """
        Detect unstable dependencies in the project.
        """
        min_dependencies = self.thresholds.get('MIN_DEPENDENCIES', 5)  # Minimum dependencies to consider
        excluded_patterns = {'test_', 'setup_', '__init__'}  # Patterns to exclude

        for node in self.module_dependencies.nodes():
            if any(pattern in node for pattern in excluded_patterns):
                continue

            in_degree = self.module_dependencies.in_degree(node)
            out_degree = self.module_dependencies.out_degree(node)
            total_dependencies = in_degree + out_degree

            if total_dependencies >= min_dependencies:
                instability = out_degree / total_dependencies
                if instability > self.thresholds['UNSTABLE_DEPENDENCY_THRESHOLD']:
                    self.add_smell(
                        "Unstable Dependency",
                        f"Module '{node}' has high instability ({instability:.2f}) " +
                        f"with {out_degree} outgoing and {in_degree} incoming dependencies",
                        self.file_paths.get(node, "Unknown"),
                        node
                    )

    def print_report(self):
        """
        Print a report of all detected architectural smells.

        If no smells are detected, it prints a message indicating so.
        """
        if not self.architectural_smells:
            print("No architectural smells detected.")
        else:
            print("Detected Architectural Smells:")
            for smell in self.architectural_smells:
                print(f"- {smell}")

    def is_entry_point(self, ast_tree):
        """Check if file is likely an entry point."""
        has_main_guard = False
        has_argparse = False

        # Check for main guard
        for node in ast_tree.body:
            if isinstance(node, ast.If):
                # Check if condition is __name__ == '__main__'
                if isinstance(node.test, ast.Compare):
                    if (
                        len(node.test.ops) == 1 and
                        isinstance(node.test.ops[0], ast.Eq) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == '__name__' and
                        len(node.test.comparators) == 1 and
                        isinstance(node.test.comparators[0], ast.Constant) and
                        node.test.comparators[0].value == '__main__'
                    ):
                        has_main_guard = True

        # Check for argparse usage
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'argparse'):
                        has_argparse = True

        return has_main_guard or has_argparse

    def _find_project_root(self, start_path):
        """
        Find the project root by scanning upwards for common project files.

        Args:
            start_path (str): Starting directory or file to scan from

        Returns:
            str: Path to project root, or start_path if not found
        """
        if os.path.isfile(start_path):
            current = os.path.dirname(os.path.abspath(start_path))
        else:
            current = os.path.abspath(start_path)

        project_indicators = ['pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt', 'Pipfile', 'poetry.lock']

        while current != os.path.dirname(current):  # Stop at filesystem root
            for indicator in project_indicators:
                if os.path.exists(os.path.join(current, indicator)):
                    return current
            current = os.path.dirname(current)

        # Fallback to original directory if no project root found
        return os.path.dirname(start_path) if os.path.isfile(start_path) else start_path

    def _find_import_for_symbol(self, symbol, local_imports):
        """
        Find the import that corresponds to a symbol.

        Args:
            symbol (str): The symbol to find the import for.
            local_imports (list): List of (import_name, line_number) tuples for imported modules

        Returns:
            str: The full import path if found, None otherwise.
        """
        for import_name, line_no in local_imports:
            # TODO: Think about whether we want to match on the last part of the imported name.
            #       This feels like it could match on the wrong thing. [fastfedora 21.Oct.25]
            # Check if the variable name matches the imported name or its last part
            if symbol == import_name or symbol == import_name.split('.')[-1]:
                return import_name
        return None

    def _resolve_ast_node(self, node):
        """
        Resolve any AST node to a string representation.

        Args:
            node (ast.AST): The AST node to resolve.

        Returns:
            str: The string representation of the AST node.

        Examples:
            ast.Name(id='print') -> "print"
            ast.Attribute(value=ast.Name(id='print'), attr='upper') -> "print.upper"
            ast.Call(
                func=ast.Name(id='print'),
                args=[ast.Constant(value='hello')]
            ) -> "print()"
            ast.Subscript(
                value=ast.Name(id='results'),
                slice=ast.Index(value=ast.Constant(value='errors'))
            ) -> "results['errors']"
            ast.Constant(value='hello') -> "<Constant>"
        """
        if isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Attribute):
            return f"{self._resolve_ast_node(node.value)}.{node.attr}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}()"
            elif isinstance(node.func, ast.Attribute):
                return f"{self._resolve_ast_node(node.func.value)}.{node.func.attr}()"
            else:
                raise ValueError(f"Unsupported function node type: {type(node.func)}")

        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Index):  # Python < 3.9
                key = self._resolve_ast_node(node.slice.value)
            else:
                key = self._resolve_ast_node(node.slice)
            return f"{self._resolve_ast_node(node.value)}[{key}]"

        elif isinstance(node, ast.Constant):
            return repr(node.value)

        else:
            # For other complex types, use the AST node type name
            node_type = type(node).__name__
            return f"<{node_type}>"

def analyze_architecture(directory_path, config_path):
    """
    Analyze the architecture of a Python project and detect architectural smells.

    Args:
        directory_path (str): The path to the directory containing the Python project to analyze.
        config_path (str): The path to the configuration file containing smell detection thresholds.
    """
    detector = ArchitecturalSmellDetector(config_path)
    detector.detect_smells(directory_path)
    detector.print_report()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect architectural smells in Python code.")
    parser.add_argument("directory", help="Directory path to analyze")
    parser.add_argument("--config", default="code_quality_config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    analyze_architecture(args.directory, args.config)