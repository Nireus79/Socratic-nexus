"""Tests for API documentation generation module."""

from socratic_nexus.documentation import (
    ParameterDoc,
    EndpointDoc,
    APIDocumentation,
    APIDocumentationGenerator,
)


class TestParameterDoc:
    """Test ParameterDoc dataclass."""

    def test_parameter_doc_creation(self):
        """Test creating a parameter documentation."""
        param = ParameterDoc(
            name="query",
            type_name="str",
            description="Search query",
        )
        assert param.name == "query"
        assert param.type_name == "str"
        assert param.description == "Search query"
        assert param.required is True
        assert param.default is None

    def test_parameter_doc_with_default(self):
        """Test parameter doc with default value."""
        param = ParameterDoc(
            name="limit",
            type_name="int",
            description="Max results",
            required=False,
            default="10",
        )
        assert param.required is False
        assert param.default == "10"

    def test_parameter_doc_with_enum(self):
        """Test parameter doc with enum values."""
        param = ParameterDoc(
            name="status",
            type_name="str",
            description="Status filter",
            enum_values=["active", "inactive", "pending"],
        )
        assert param.enum_values == ["active", "inactive", "pending"]

    def test_parameter_doc_to_dict(self):
        """Test converting parameter doc to dictionary."""
        param = ParameterDoc(
            name="query",
            type_name="str",
            description="Search query",
            enum_values=["option1", "option2"],
        )
        result = param.to_dict()
        assert result["name"] == "query"
        assert result["type"] == "str"
        assert result["description"] == "Search query"
        assert result["required"] is True
        assert result["enum"] == ["option1", "option2"]

    def test_parameter_doc_to_dict_no_enum(self):
        """Test parameter doc to dict without enum."""
        param = ParameterDoc(
            name="count",
            type_name="int",
            description="Item count",
        )
        result = param.to_dict()
        assert result["enum"] is None


class TestEndpointDoc:
    """Test EndpointDoc dataclass."""

    def test_endpoint_doc_creation(self):
        """Test creating endpoint documentation."""
        endpoint = EndpointDoc(
            name="search",
            summary="Search items",
            description="Search for items by query",
        )
        assert endpoint.name == "search"
        assert endpoint.summary == "Search items"
        assert endpoint.description == "Search for items by query"
        assert endpoint.method_type == "method"
        assert len(endpoint.parameters) == 0

    def test_endpoint_doc_with_parameters(self):
        """Test endpoint doc with parameters."""
        param = ParameterDoc(
            name="query",
            type_name="str",
            description="Search query",
        )
        endpoint = EndpointDoc(
            name="search",
            summary="Search items",
            description="Search items",
            parameters=[param],
        )
        assert len(endpoint.parameters) == 1
        assert endpoint.parameters[0].name == "query"

    def test_endpoint_doc_with_exceptions(self):
        """Test endpoint doc with exceptions."""
        endpoint = EndpointDoc(
            name="fetch",
            summary="Fetch data",
            description="Fetch data",
            raises=["ValueError", "TimeoutError"],
        )
        assert endpoint.raises == ["ValueError", "TimeoutError"]

    def test_endpoint_doc_deprecated(self):
        """Test deprecated endpoint."""
        endpoint = EndpointDoc(
            name="old_method",
            summary="Old method",
            description="This is deprecated",
            deprecated=True,
            deprecated_message="Use new_method instead",
        )
        assert endpoint.deprecated is True
        assert endpoint.deprecated_message == "Use new_method instead"

    def test_endpoint_doc_to_dict(self):
        """Test converting endpoint doc to dictionary."""
        param = ParameterDoc(
            name="id",
            type_name="int",
            description="Resource ID",
        )
        endpoint = EndpointDoc(
            name="get_item",
            summary="Get item",
            description="Get item by ID",
            parameters=[param],
            returns="Item data",
            return_type="dict",
            raises=["NotFoundError"],
        )
        result = endpoint.to_dict()
        assert result["name"] == "get_item"
        assert result["summary"] == "Get item"
        assert len(result["parameters"]) == 1
        assert result["returns"] == "Item data"
        assert result["return_type"] == "dict"
        assert "NotFoundError" in result["raises"]


class TestAPIDocumentation:
    """Test APIDocumentation dataclass."""

    def test_api_documentation_creation(self):
        """Test creating API documentation."""
        api_doc = APIDocumentation(
            title="Search API",
            version="1.0.0",
            description="API for searching items",
        )
        assert api_doc.title == "Search API"
        assert api_doc.version == "1.0.0"
        assert api_doc.description == "API for searching items"
        assert len(api_doc.endpoints) == 0
        assert api_doc.base_url == ""

    def test_api_documentation_with_endpoints(self):
        """Test API documentation with endpoints."""
        endpoint = EndpointDoc(
            name="search",
            summary="Search",
            description="Search items",
        )
        api_doc = APIDocumentation(
            title="API",
            version="1.0.0",
            description="Test API",
            endpoints=[endpoint],
        )
        assert len(api_doc.endpoints) == 1
        assert api_doc.endpoints[0].name == "search"

    def test_api_documentation_with_authentication(self):
        """Test API documentation with authentication."""
        api_doc = APIDocumentation(
            title="API",
            version="1.0.0",
            description="Test API",
            authentication={"type": "Bearer", "scheme": "HTTP"},
        )
        assert api_doc.authentication["type"] == "Bearer"

    def test_api_documentation_with_schemas(self):
        """Test API documentation with schemas."""
        schemas = {
            "Item": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
            }
        }
        api_doc = APIDocumentation(
            title="API",
            version="1.0.0",
            description="Test API",
            schemas=schemas,
        )
        assert "Item" in api_doc.schemas
        assert api_doc.schemas["Item"]["type"] == "object"

    def test_api_documentation_to_dict(self):
        """Test converting API documentation to dictionary."""
        endpoint = EndpointDoc(
            name="list",
            summary="List items",
            description="List all items",
        )
        api_doc = APIDocumentation(
            title="API",
            version="2.0.0",
            description="Test API",
            base_url="https://api.example.com",
            endpoints=[endpoint],
            tags=["search", "items"],
        )
        result = api_doc.to_dict()
        assert result["title"] == "API"
        assert result["version"] == "2.0.0"
        assert result["base_url"] == "https://api.example.com"
        assert len(result["endpoints"]) == 1
        assert "search" in result["tags"]


class TestAPIDocumentationGenerator:
    """Test APIDocumentationGenerator class."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        gen = APIDocumentationGenerator(
            title="Test API",
            version="1.0.0",
            description="Test description",
        )
        assert gen.title == "Test API"
        assert gen.version == "1.0.0"
        assert gen.description == "Test description"
        assert len(gen.endpoints) == 0

    def test_document_function(self):
        """Test documenting a simple function."""

        def sample_func(x: int, y: str) -> str:
            """Add description.

            Args:
                x: An integer value
                y: A string value

            Returns:
                Combined result
            """
            return f"{x}-{y}"

        gen = APIDocumentationGenerator("API", "1.0.0")
        endpoint = gen.document_method(sample_func)

        assert endpoint.name == "sample_func"
        assert endpoint.summary == "Add description."
        assert endpoint.return_type == "str"

    def test_document_method_with_class(self):
        """Test documenting a method with class name."""

        def method(self, name: str) -> None:
            """Process name.

            Args:
                name: The name to process
            """
            pass

        gen = APIDocumentationGenerator("API", "1.0.0")
        endpoint = gen.document_method(method, class_name="Processor")

        assert endpoint.name == "Processor.method"

    def test_document_class(self):
        """Test documenting all public methods in a class."""

        class SampleClass:
            def method1(self) -> None:
                """First method."""
                pass

            def method2(self) -> None:
                """Second method."""
                pass

            def _private_method(self) -> None:
                """Private method."""
                pass

        gen = APIDocumentationGenerator("API", "1.0.0")
        gen.document_class(SampleClass)

        # Should document public methods only
        public_methods = [e.name for e in gen.endpoints]
        assert any("method1" in name for name in public_methods)
        assert any("method2" in name for name in public_methods)
        # Private methods should not be documented
        assert not any("_private_method" in name for name in public_methods)

    def test_extract_parameters_with_types(self):
        """Test parameter extraction with type annotations."""

        def func(name: str, count: int = 5) -> None:
            """Test function.

            Args:
                name: The name
                count: The count
            """
            pass

        gen = APIDocumentationGenerator("API", "1.0.0")
        endpoint = gen.document_method(func)

        assert len(endpoint.parameters) == 2
        param_names = [p.name for p in endpoint.parameters]
        assert "name" in param_names
        assert "count" in param_names

        # Check required vs optional
        name_param = next(p for p in endpoint.parameters if p.name == "name")
        count_param = next(p for p in endpoint.parameters if p.name == "count")
        assert name_param.required is True
        assert count_param.required is False
        assert count_param.default == "5"

    def test_document_method_with_docstring_parsing(self):
        """Test method with complete docstring."""

        def search(query: str, limit: int = 10) -> dict:
            """Search for items.

            This is a more detailed description of the search function.
            It searches across multiple sources.

            Args:
                query: The search query string
                limit: Maximum results to return

            Returns:
                Dictionary with results and metadata

            Raises:
                ValueError: If query is empty
                TimeoutError: If search takes too long
            """
            return {}

        gen = APIDocumentationGenerator("API", "1.0.0")
        endpoint = gen.document_method(search)

        assert endpoint.summary == "Search for items."
        assert "more detailed description" in endpoint.description
        assert len(endpoint.parameters) == 2
        assert endpoint.returns == "Dictionary with results and metadata"
        assert "ValueError" in endpoint.raises
        assert "TimeoutError" in endpoint.raises

    def test_document_method_without_docstring(self):
        """Test documenting method without docstring."""

        def undocumented_func(x):
            return x

        gen = APIDocumentationGenerator("API", "1.0.0")
        endpoint = gen.document_method(undocumented_func)

        assert endpoint.summary == ""
        assert endpoint.description == ""
        assert len(endpoint.parameters) >= 0

    def test_generator_collects_endpoints(self):
        """Test that generator collects all documented endpoints."""

        def func1() -> None:
            """First function."""
            pass

        def func2() -> None:
            """Second function."""
            pass

        gen = APIDocumentationGenerator("API", "1.0.0")
        gen.document_method(func1)
        gen.document_method(func2)

        assert len(gen.endpoints) == 2
        names = [e.name for e in gen.endpoints]
        assert "func1" in names
        assert "func2" in names

    def test_parameter_type_extraction(self):
        """Test parameter type extraction from annotations."""

        def typed_func(text: str, items: list, count: int) -> None:
            """Process data.

            Args:
                text: Text to process
                items: List of items
                count: Number of items
            """
            pass

        gen = APIDocumentationGenerator("API", "1.0.0")
        endpoint = gen.document_method(typed_func)

        params = {p.name: p.type_name for p in endpoint.parameters}
        assert params["text"] == "str"
        assert params["count"] == "int"

    def test_return_type_extraction(self):
        """Test return type extraction."""

        def returns_dict() -> dict:
            """Return a dictionary."""
            return {}

        def returns_list() -> list:
            """Return a list."""
            return []

        gen = APIDocumentationGenerator("API", "1.0.0")
        dict_endpoint = gen.document_method(returns_dict)
        list_endpoint = gen.document_method(returns_list)

        assert dict_endpoint.return_type == "dict"
        assert list_endpoint.return_type == "list"
