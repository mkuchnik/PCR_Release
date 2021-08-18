class MultiviewIterator:
    """An iterator capable of supplying multiple views into the same data."""
    def __init__(self, data):
        """Take a finite list of in-memory data"""
        self.data = data

    def get_iterator(self, view_name: str):
        return self.views[view_name]

    def register_view(self, view_name: str, view_function):
        """Maps from name to a function: dataset -> dataset"""
        self.views[view_name] = view_function
