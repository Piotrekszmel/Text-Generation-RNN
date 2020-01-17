from abc import ABCMeta, abstractmethod, ABC
from frozendict import frozendict
import sys
sys.path.append('..')


class ResourceManager(metaclass=ABCMeta):
  def __init__(self):
    self.wv_filename = ""
    self.parsed_filename = ""

  @abstractmethod
  def write(self):
    """
    parse the raw file/files and write the data to disk
    """
    pass

  @abstractmethod
  def read(self):
    """
    read the parsed file from disk
    """
    pass
    
  def read_hashable(self):
    return frozendict(self.read())