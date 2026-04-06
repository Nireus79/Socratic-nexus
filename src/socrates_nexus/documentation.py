"""Automatic API documentation generation for Socrates Nexus."""

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ParameterDoc:
    """Documentation for a parameter."""

    name: str
    type_name: str
    description: str
    required: bool = True
    default: Optional[str] = None
    enum_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type_name,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "enum": self.enum_values if self.enum_values else None,
        }


@dataclass
class EndpointDoc:
    """Documentation for an API endpoint/method."""

    name: str
    summary: str
    description: str
    method_type: str = "method"  # 'method', 'function'
    parameters: List[ParameterDoc] = field(default_factory=list)
    returns: str = ""
    return_type: str = ""
    raises: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    since_version: str = ""
    deprecated: bool = False
    deprecated_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "method_type": self.method_type,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns,
            "return_type": self.return_type,
            "raises": self.raises,
            "examples": self.examples,
            "since_version": self.since_version,
            "deprecated": self.deprecated,
            "deprecated_message": self.deprecated_message,
        }


@dataclass
class APIDocumentation:
    """Complete API documentation."""

    title: str
    version: str
    description: str
    base_url: str = ""
    endpoints: List[EndpointDoc] = field(default_factory=list)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    authentication: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "version": self.version,
            "description": self.description,
            "base_url": self.base_url,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "schemas": self.schemas,
            "authentication": self.authentication,
            "tags": self.tags,
        }


class APIDocumentationGenerator:
    """Generates API documentation from Python code."""

    def __init__(self, title: str, version: str, description: str = ""):
        """
        Initialize documentation generator.

        Args:
            title: API title
            version: API version
            description: API description
        """
        self.title = title
        self.version = version
        self.description = description
        self.endpoints: List[EndpointDoc] = []
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def document_class(self, cls: Type) -> None:
        """
        Auto-document all public methods of a class.

        Args:
            cls: Class to document
        """
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue

            try:
                method = getattr(cls, method_name)
                if callable(method):
                    self.document_method(method, class_name=cls.__name__)
            except Exception as e:
                self.logger.warning(f"Could not document {cls.__name__}.{method_name}: {e}")

    def document_method(
        self,
        method: Callable,
        class_name: Optional[str] = None,
    ) -> EndpointDoc:
        """
        Document a single method.

        Args:
            method: Method to document
            class_name: Optional class name

        Returns:
            EndpointDoc with extracted documentation
        """
        name = method.__name__
        if class_name:
            name = f"{class_name}.{name}"

        # Extract docstring
        docstring = inspect.getdoc(method) or ""
        lines = docstring.split("\n")
        summary = lines[0] if lines else ""
        description = "\n".join(lines[2:]) if len(lines) > 2 else ""

        # Extract parameters
        sig = inspect.signature(method)
        parameters = self._extract_parameters(sig, docstring)

        # Extract return type
        return_type = self._extract_return_type(sig)
        returns = self._extract_returns_description(docstring)

        # Extract exceptions
        raises = self._extract_raises(docstring)

        endpoint = EndpointDoc(
            name=name,
            summary=summary,
            description=description,
            parameters=parameters,
            return_type=return_type,
            returns=returns,
            raises=raises,
        )

        self.endpoints.append(endpoint)
        return endpoint

    def _extract_parameters(
        self,
        sig: inspect.Signature,
        docstring: str,
    ) -> List[ParameterDoc]:
        """Extract parameters from signature and docstring."""
        parameters = []
        param_docs = self._parse_parameter_docs(docstring)

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            type_annotation = (
                param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "Any"
            )

            param_doc = ParameterDoc(
                name=param_name,
                type_name=type_annotation,
                description=param_docs.get(param_name, ""),
                required=param.default == inspect.Parameter.empty,
                default=str(param.default) if param.default != inspect.Parameter.empty else None,
            )

            parameters.append(param_doc)

        return parameters

    def _extract_return_type(self, sig: inspect.Signature) -> str:
        """Extract return type from signature."""
        if sig.return_annotation != inspect.Signature.empty:
            return_type = sig.return_annotation
            if hasattr(return_type, "__name__"):
                return return_type.__name__
            return str(return_type)
        return "Any"

    def _extract_returns_description(self, docstring: str) -> str:
        """Extract returns description from docstring."""
        lines = docstring.split("\n")
        in_returns = False
        returns_lines = []

        for line in lines:
            if line.strip().startswith("Returns:"):
                in_returns = True
                continue
            elif line.strip().startswith(("Args:", "Raises:", "Yields:", "Note:")):
                in_returns = False
            elif in_returns and line.strip():
                returns_lines.append(line.strip())

        return " ".join(returns_lines)

    def _extract_raises(self, docstring: str) -> List[str]:
        """Extract exception types from docstring."""
        lines = docstring.split("\n")
        raises = []
        in_raises = False

        for line in lines:
            if line.strip().startswith("Raises:"):
                in_raises = True
                continue
            elif line.strip().startswith(("Args:", "Returns:", "Note:")):
                in_raises = False
            elif in_raises and line.strip() and ":" in line:
                exception_type = line.split(":")[0].strip()
                raises.append(exception_type)

        return raises

    def _parse_parameter_docs(self, docstring: str) -> Dict[str, str]:
        """Parse parameter documentation from docstring."""
        docs = {}
        lines = docstring.split("\n")
        in_args = False

        for i, line in enumerate(lines):
            if line.strip().startswith("Args:"):
                in_args = True
                continue
            elif line.strip().startswith(("Returns:", "Raises:")):
                in_args = False
            elif in_args and line.strip() and ":" in line:
                parts = line.split(":", 1)
                param_name = parts[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                docs[param_name] = param_desc

        return docs

    def add_schema(
        self,
        name: str,
        schema: Dict[str, Any],
    ) -> None:
        """
        Add a schema definition.

        Args:
            name: Schema name
            schema: Schema definition
        """
        self.schemas[name] = schema

    def to_openapi(self) -> Dict[str, Any]:
        """
        Generate OpenAPI 3.0 specification.

        Returns:
            OpenAPI spec dictionary
        """
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "paths": {},
            "components": {
                "schemas": self.schemas,
            },
        }

        # Convert endpoints to paths
        for endpoint in self.endpoints:
            path_key = f"/api/{endpoint.name.lower().replace('.', '/')}"

            openapi_spec["paths"][path_key] = {
                "post": {
                    "summary": endpoint.summary,
                    "description": endpoint.description,
                    "parameters": self._convert_to_openapi_params(endpoint.parameters),
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"result": {"type": endpoint.return_type}},
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad Request"},
                        "500": {"description": "Internal Server Error"},
                    },
                }
            }

        return openapi_spec

    def _convert_to_openapi_params(
        self,
        parameters: List[ParameterDoc],
    ) -> List[Dict[str, Any]]:
        """Convert ParameterDoc to OpenAPI format."""
        openapi_params = []

        for param in parameters:
            openapi_param = {
                "name": param.name,
                "in": "query",
                "required": param.required,
                "schema": {
                    "type": param.type_name.lower(),
                },
            }

            if param.description:
                openapi_param["description"] = param.description

            if param.enum_values:
                openapi_param["schema"]["enum"] = param.enum_values

            openapi_params.append(openapi_param)

        return openapi_params

    def to_markdown(self) -> str:
        """
        Generate Markdown documentation.

        Returns:
            Markdown documentation string
        """
        lines = [
            f"# {self.title}",
            "",
            f"**Version:** {self.version}",
            "",
            self.description,
            "",
            "## Endpoints",
            "",
        ]

        for endpoint in self.endpoints:
            lines.append(f"### {endpoint.name}")
            lines.append("")
            lines.append(f"**Summary:** {endpoint.summary}")
            lines.append("")

            if endpoint.description:
                lines.append("**Description:**")
                lines.append("")
                lines.append(endpoint.description)
                lines.append("")

            if endpoint.parameters:
                lines.append("**Parameters:**")
                lines.append("")
                for param in endpoint.parameters:
                    required = "required" if param.required else "optional"
                    lines.append(
                        f"- `{param.name}` ({param.type_name}) {required}: {param.description}"
                    )
                lines.append("")

            if endpoint.return_type:
                lines.append(f"**Returns:** {endpoint.return_type}")
                if endpoint.returns:
                    lines.append(f" - {endpoint.returns}")
                lines.append("")

            if endpoint.raises:
                lines.append("**Raises:**")
                lines.append("")
                for exc in endpoint.raises:
                    lines.append(f"- {exc}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """
        Generate JSON documentation.

        Returns:
            JSON documentation string
        """
        doc_dict = {
            "title": self.title,
            "version": self.version,
            "description": self.description,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "schemas": self.schemas,
        }
        return json.dumps(doc_dict, indent=2)

    def export(self, filepath: str, format: str = "markdown") -> None:
        """
        Export documentation to file.

        Args:
            filepath: Output file path
            format: Output format ('markdown', 'json', 'openapi')
        """
        try:
            if format == "markdown":
                content = self.to_markdown()
            elif format == "json":
                content = self.to_json()
            elif format == "openapi":
                content = json.dumps(self.to_openapi(), indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            with open(filepath, "w") as f:
                f.write(content)

            self.logger.info(f"Documentation exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export documentation: {e}")
            raise
