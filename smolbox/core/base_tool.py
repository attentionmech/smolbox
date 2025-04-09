# smolbox/tools/base_tool.py

class BaseTool:
    def __init__(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the init() method.")

    def run(self):
        raise NotImplementedError("Subclasses must implement the run() method.")
