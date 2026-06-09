#tool registry: holds directory of every tool the agent can use. 
#tools register themselves via the @registry.tool decorator.
import inspect
import json

class ToolRegistry:
    def __init__(self):
        #"phonebook" analogy
        #keys are tool names (strings), values are actual functions
        self._tools = {}
    
    def tool(self, fn):
        # decorator that registers a function as an agent tool
        #python calls registry.tool(fn) automatically, passing the function in
        #stores, then returns it unchanged so it still works normally
        self._tools[fn.__name__] = fn
        print(f"[registry] registered tool: {fn.__name__}")
        return fn
    def get(self, name:str):
        #look up a tool by name, return none if not found
        return self._tools.get(name)
    def get_schemas(self) -> list:
        #builds the list of tool defintions claude needs
        #needs name, descript., input schema
        schemas = []
        for name, fn in self._tools.items():
            schema = _build_schema(fn)
            schemas.append(schema)
        return schemas
    def list_tools(self) -> list[str]:
        #returns a list of all registered tool names
        return list(self._tools.keys())
def _build_schema(fn) -> dict:
        #inspects a function and builds a claude-compatible tool schema from it
        sig = inspect.signature(fn)
        properties = {}
        required = []
        type_map = {
            str:"string",
            int:"integer",
            float:"number",
            bool:"boolean",
            list:"array",
            dict:"object"
        }
        
        for param_name, param in sig.parameters.items():
            annotation = param.annotation

            json_type = type_map.get(annotation,"string")
            properties[param_name] = {
                "type":json_type,
                "description": f"{param_name} parameter",
            }
            #if no default value, the argument is req.
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or f"{fn.__name__} tool",
            "input_schema": {
                "type":"object",
                "properties": properties,
                "required": required
            },
        }
registry = ToolRegistry()