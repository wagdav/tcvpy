""" DataSource """


class DataSource(object):
    """ Abstract class to represent a data source. """

    def close(self):
        """ Close the DataSource """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
