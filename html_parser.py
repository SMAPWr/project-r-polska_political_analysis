from dataclasses import dataclass, field
from typing import Type
from functools import reduce
from html.parser import HTMLParser

empty_tags = ['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 
              'keygen', 'link', 'meta', 'param', 'source', 'track', 'wbr']

class NodeList:
    def __init__(self, nodes):
        self.nodes = nodes
        
    def find(self, tag='', attrs=[], data=[]):
        results = [c.find(tag, attrs, data) for c in self.nodes]
        return NodeList([item for sublist in results for item in sublist.nodes])
    
    def contain(self, tag='', attrs=[], data=[]):
        results = [c.contain(tag, attrs, data) for c in self.nodes]
        return NodeList([item for sublist in results for item in sublist.nodes])
    
    def append(self, el):
        self.nodes.append(el)
    
    def __iter__(self):
        return self.nodes.__iter__()
    
    def __getitem__(self, i):
        return self.nodes[i]
    
    def __len__(self):
        return len(self.nodes)

@dataclass
class Node:
    tag: str
    attrs: dict
    data: str
    parent: Type['Node']
    children: list = field(default_factory=list)
        
    def add_child(self, child: Type['Node']):
        self.children.append(child)
    
    def find(self, tag='', attrs=[], data=[]):
        queue = self.children.copy()
        found = []
        while len(queue) > 0:
            node = queue.pop(0)
            f_1 = lambda v: v in node.data
            f_2 = lambda kv: kv[0] in node.attrs and kv[1] in node.attrs[kv[0]]
            if (tag == '' or node.tag == tag) and all(map(f_2, attrs)) and all(map(f_1, data)):
                found.append(node)
            else:
                queue += node.children
        return NodeList(found)
    
    def contain(self, tag, attrs=[], data=[]):
        nodelist = NodeList([])
        if len(self.find(tag, attrs, data)) > 0:
            nodelist.append(self)
        return nodelist
    
    
class WykopParser(HTMLParser):    
    def __init__(self):
        super().__init__()
        self.root = Node('DOCUMENT_ROOT', {}, '', None)
        self.current_node = self.root
         
    def handle_starttag(self, tag, attrs, emptytag=False): 
        attrs = {key: value for key, value in attrs}
        new_node = Node(tag, attrs, '', self.current_node)
        self.current_node.add_child(new_node)
        self.current_node = new_node
        
        if not emptytag and tag in empty_tags:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self.current_node = self.current_node.parent

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs, True)
        self.handle_endtag(tag)
        
    def handle_data(self, data):
        self.current_node.data += data    