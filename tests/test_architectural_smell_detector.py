import pytest
import ast
from code_quality_analyzer.architectural_smell_detector import ArchitecturalSmellDetector
from code_quality_analyzer.config_handler import ConfigHandler

@pytest.fixture
def config_handler():
    return ConfigHandler('code_quality_config.yaml')

@pytest.fixture
def architectural_smell_detector(config_handler):
    thresholds = config_handler.get_thresholds('architectural_smells')
    return ArchitecturalSmellDetector(thresholds)

def test_detect_god_object(architectural_smell_detector, tmp_path):
    test_file = tmp_path / "god_object.py"
    test_file.write_text("\n".join([f"def func{i}(): pass" for i in range(26)]))

    architectural_smell_detector.detect_smells(str(tmp_path))
    assert any("God Object" in smell.name for smell in architectural_smell_detector.architectural_smells)


def test_detect_scattered_functionality(architectural_smell_detector, tmp_path):
    for i in range(3):
        module = tmp_path / f"module{i}.py"
        module.write_text("def scattered_function(): pass")

    architectural_smell_detector.detect_smells(str(tmp_path))
    assert any("Scattered Functionality" in smell.name for smell in architectural_smell_detector.architectural_smells)

def test_detect_redundant_abstraction(architectural_smell_detector, tmp_path):
    module1 = tmp_path / "module1.py"
    module2 = tmp_path / "module2.py"
    content = "\n".join([f"def func{i}(): pass" for i in range(5)])
    module1.write_text(content)
    module2.write_text(content)

    architectural_smell_detector.detect_smells(str(tmp_path))
    assert any("Redundant Abstraction" in smell.name for smell in architectural_smell_detector.architectural_smells)

def test_detect_improper_api_usage(architectural_smell_detector, tmp_path):
    test_file = tmp_path / "improper_api_usage.py"
    test_file.write_text("\n".join([f"api.method1()" for _ in range(10)]))

    architectural_smell_detector.detect_smells(str(tmp_path))
    assert any("Improper API Usage" in smell.name for smell in architectural_smell_detector.architectural_smells)


def test_resolve_api_call_none_context(architectural_smell_detector):
    """Test direct function calls with no object context."""
    tree = ast.parse("print('hello')")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("print", None, "none")

    tree = ast.parse("len(my_list)")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("len", None, "none")


def test_resolve_api_call_imported_symbol_context(architectural_smell_detector):
    """Test calls to imported modules/classes."""
    tree = ast.parse("sqlite3.connect('db')")
    call_node = tree.body[0].value  # The Call node
    local_imports = [('sqlite3', 1)]
    result = architectural_smell_detector.resolve_api_call(call_node.func, local_imports)
    assert result == ("connect", "sqlite3", "imported-symbol")

    tree = ast.parse("requests.get('url')")
    call_node = tree.body[0].value  # The Call node
    local_imports = [('requests', 1)]
    result = architectural_smell_detector.resolve_api_call(call_node.func, local_imports)
    assert result == ("get", "requests", "imported-symbol")


def test_resolve_api_call_local_symbol_context(architectural_smell_detector):
    """Test calls to local variables or other symbols."""
    tree = ast.parse("cursor.execute('sql')")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("execute", "cursor", "local-symbol")

    tree = ast.parse("my_list.append('item')")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("append", "my_list", "local-symbol")


def test_resolve_api_call_attribute_chain_context(architectural_smell_detector):
    """Test chained attribute access."""
    tree = ast.parse("obj.subobj.method()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("method", "obj.subobj", "attribute-chain")

    tree = ast.parse("request.headers.get()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("get", "request.headers", "attribute-chain")


def test_resolve_api_call_call_chain_context(architectural_smell_detector):
    """Test method calls on function return values."""
    tree = ast.parse("hashlib.sha256().hexdigest()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("hexdigest", "hashlib.sha256()", "call-chain")

    tree = ast.parse("datetime.now().strftime('%Y')")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("strftime", "datetime.now()", "call-chain")


def test_resolve_api_call_subscript_context(architectural_smell_detector):
    """Test method calls on subscripted objects."""
    tree = ast.parse("results['errors'].append()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("append", "results['errors']", "subscript")

    tree = ast.parse("data['users'].extend()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("extend", "data['users']", "subscript")


def test_resolve_api_call_expression_context(architectural_smell_detector):
    """Test method calls on complex expressions."""
    tree = ast.parse("(password + salt).encode()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("encode", "<BinOp>", "expression")

    tree = ast.parse("(obj1 + obj2).method()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("method", "<BinOp>", "expression")


def test_resolve_api_call_constant_context(architectural_smell_detector):
    """Test method calls on constants."""
    # Test string constant method call
    tree = ast.parse("'password123'.encode()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("encode", "<Constant>", "expression")

    # Test join method on string constant
    tree = ast.parse("', '.join(my_list)")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("join", "<Constant>", "expression")

    # Test method on numeric constant
    tree = ast.parse("(42).bit_length()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("bit_length", "<Constant>", "expression")


def test_resolve_api_call_unknown_context(architectural_smell_detector):
    """Test unrecognized patterns."""
    # Test with a non-AST node (should not happen in practice)
    class MockNode:
        attr = "unknown_func"

    func_node = MockNode()

    # Should raise ValueError for unsupported node types
    with pytest.raises(ValueError, match="Unsupported function node type"):
        architectural_smell_detector.resolve_api_call(func_node, [])


def test_resolve_api_call_import_resolution_edge_cases(architectural_smell_detector):
    """Test edge cases in import resolution."""
    # Test with multiple imports
    tree = ast.parse("requests.get()")
    call_node = tree.body[0].value  # The Call node
    local_imports = [('sqlite3', 1), ('requests', 2), ('os', 3)]
    result = architectural_smell_detector.resolve_api_call(call_node.func, local_imports)
    assert result == ("get", "requests", "imported-symbol")

    # Test with no matching import
    tree = ast.parse("unknown_module.method()")
    call_node = tree.body[0].value  # The Call node
    local_imports = [('sqlite3', 1), ('requests', 2)]
    result = architectural_smell_detector.resolve_api_call(call_node.func, local_imports)
    assert result == ("method", "unknown_module", "local-symbol")

    # Test with partial import name match
    tree = ast.parse("connect.execute()")
    call_node = tree.body[0].value  # The Call node
    local_imports = [('sqlite3.connect', 1)]
    result = architectural_smell_detector.resolve_api_call(call_node.func, local_imports)
    assert result == ("execute", "sqlite3.connect", "imported-symbol")


def test_resolve_api_call_complex_attribute_chains(architectural_smell_detector):
    """Test deeply nested attribute chains."""
    tree = ast.parse("obj.subobj.subsubobj.method()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("method", "obj.subobj.subsubobj", "attribute-chain")


def test_resolve_api_call_complex_call_chains(architectural_smell_detector):
    """Test complex function call chains."""
    tree = ast.parse("obj.method().submethod()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("submethod", "obj.method()", "call-chain")


def test_resolve_api_call_complex_subscript_chains(architectural_smell_detector):
    """Test complex subscript chains."""
    tree = ast.parse("data['users']['admin'].append()")
    call_node = tree.body[0].value  # The Call node
    result = architectural_smell_detector.resolve_api_call(call_node.func, [])
    assert result == ("append", "data['users']['admin']", "subscript")

