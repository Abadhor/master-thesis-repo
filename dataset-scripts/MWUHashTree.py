

class Node():
  
  def __init__(self):
    self.dictionary = {}
    self.value = None
  
  def add(self, token_list, value):
    """ If the last add operation adds something, return True """
    first = token_list[0]
    if first not in self.dictionary:
      self.dictionary[first] = Node()
      if len(token_list) <= 1:
        self.dictionary[first].value = value
        return True
      else:
        return self.dictionary[first].add(token_list[1:], value)
    if len(token_list) <= 1:
      # the token_list is already part of the tree
      # check if it has already been added exactly like this
      if not self.dictionary[first].value:
        self.dictionary[first].value = value
        return True
      else:
        return False
    else:
      return self.dictionary[first].add(token_list[1:], value)
  
  def getAll(self, token_list, cur_tokens, return_list):
    first = token_list[0]
    cur_tokens.append(first)
    if first not in self.dictionary:
      return return_list
    if self.dictionary[first].value:
      return_list.append(cur_tokens.copy())
    if len(token_list) <= 1:
      return return_list
    else:
      return self.dictionary[first].getAll(token_list[1:], cur_tokens, return_list)
  
  def __contains__(self, token_list):
    first = token_list[0] 
    if first not in self.dictionary:
      return False
    if len(token_list) > 1:
      return token_list[1:] in self.dictionary[first]
    if not self.dictionary[first].value:
      return False
    return True
  
  def __getitem__(self, token_list):
    first = token_list[0] 
    if first not in self.dictionary:
      raise KeyError(token_list)
    if len(token_list) > 1:
      return self.dictionary[first][token_list[1:]]
    if not self.dictionary[first].value:
      raise KeyError(token_list)
    return self.dictionary[first].value
  
  def items(self, cur_tokens, return_list):
    for item in self.dictionary.items():
      token = item[0]
      node = item[1]
      copy = cur_tokens.copy()
      copy.append(token)
      if node.value:
        return_list.append((copy, node.value))
      node.items(copy, return_list)


class MWUHashTree():
  
  def __init__(self):
    self.max_len = 0
    self.length = 0
    self.root = Node()
  
  def __setitem__(self, key, value):
    added = self.root.add(key, value)
    if added:
      self.length += 1
    if self.max_len < len(key):
      self.max_len = len(key)
  
  def getAll(self, token_list):
    return self.root.getAll(token_list, [], [])
  
  def __contains__(self, item):
    return item in self.root
  
  def __getitem__(self, key):
    if key not in self.root:
      raise KeyError(key)
    return self.root[key]
  
  def __len__(self):
    return self.length
  
  def items(self):
    cur_tokens = []
    return_list = []
    self.root.items(cur_tokens,return_list)
    return return_list
  
  def keys(self):
    items = self.items()
    return [item[0] for item in items]
  
  def values(self):
    items = self.items()
    return [item[1] for item in items]
  