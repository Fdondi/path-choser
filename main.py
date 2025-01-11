from pathlib import Path
import pickle
import random
from typing import Optional, Union
import pandas as pd

class Node:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __str__(self):
        res = self.name
        if self.description:
            res += f" ({self.description[:100]})"
        return res

    def __repr__(self):
        return f"Node({self.name}, {self.description})"

class Leaf(Node):
    def __init__(self, name, description):
        super().__init__(name, description)

class DecisionTree(Node):
    def __init__(self, name, description = ""):
        super().__init__(name, description)
        self.children = pd.DataFrame(columns=['children', 'probabilities'])
        self.leaves = pd.Series()
    
    def print_contents(self):
        if not self.children.empty:
            print("Children:")
            print(self.children["children"])
        if not self.leaves.empty:
            print("Leaves:")
            print(self.leaves.values)

    def _normalize_probabilities(self):
        total = sum(self.children["probabilities"])
        self.children["probabilities"] = [prob/total for prob in self.children["probabilities"]]

    def add_child(self) ->  Optional[Node]:
        print("Add child!")
        name = input("Enter child name: ")
        if not name:
            print("Ok, no child!")
            return None
        description = input("Enter child description: ")
        child = DecisionTree(name, description)
        # set to the average existing probability
        # empty -> 1 // 1 -> 1,1 -> 1/2,1/2 // 1/2,1/2 -> 1/2,1/2,1/2 -> 1/3,1/3,1/3
        new_prob = 1 if self.children.empty else  1/len(self.children)
        self.children.loc[len(self.children)] = {"children": child, "probabilities": new_prob}
        self._normalize_probabilities()
        return child

    def add_leaf(self) -> bool:
        print("Add leaf!")
        name = input("Enter leaf name: ")
        if not name:
            print("Ok, no leaf!")
            return False
        description = input("Enter leaf description: ")
        self.leaves.loc[len(self.leaves)] = Leaf(name, description)
        return True

    def partition(self):
        print("Partition!")
        while len(self.children) > 1 or len(self.leaves) > 0:
            print("Currently:")
            self.print_contents()
            group_name = input("Enter group name (empty to leave): ")
            if not group_name:
                print("Empty name, leaving existing members as is")
                break
            group_description = input("Enter group description: ")
            new_child = DecisionTree(group_name, group_description)
            # select children to move to group
            group_children = input("Enter group children (space separated): ")
            group_children = [int(i) for i in group_children.split()]
            selected_children = self.children[group_children]
            new_prob = sum(selected_children['probabilities'])
            self.children.drop(group_children, inplace=True)
            new_child.children = selected_children
            new_child._normalize_probabilities()
            # select leaves to move to group
            group_leaves = input("Enter group leaves (space separated): ")
            group_leaves = [int(i) for i in group_leaves.split()]
            new_child.leaves = self.leaves[group_leaves]
            self.leaves.drop(group_leaves, inplace=True)
            # add new group
            self.children.loc[len(self.children)] = {"children": new_child, "probabilities": new_prob}

    def update_prob(self, index, rel_delta):
        delta = rel_delta * self.children.loc[index, "probabilities"]
        self.children.loc[index, "probabilities"] += delta
        self._normalize_probabilities()

    def visit(self):
        print(f"Visiting: {self}")
        self.print_contents()
        # with probability (20-n_children)/20 add a child
        if random.randint(0, 20) >= len(self.children):
           added_child = self.add_child()
           if added_child:
                added_child.visit()
                return
        # if no children, nothing to do but add a leave
        if self.children.empty or (random.randint(0, 20) >=len(self.children) + len(self.leaves)):
            added_leaf = self.add_leaf()
            if added_leaf:
                return 
        # if too many more than 10 elements, divide chidren into groups
        if random.randint(10, 30) < len(self.leaves) + len(self.children):
            self.partition()
        # actually visit
        # select one child with probability proportional to children_probabilities
        choice = 'n'
        while choice == 'n':
            selected_index = random.choices(self.children.index, weights=self.children["probabilities"], k=1)[0]
            selected_child = self.children.loc[selected_index, "children"]
            print(f"Suggesting: {selected_child}")
            choice = input("Is this ok? [(s)uper/(o)k/(n)o/]: ")
            if choice == 's':
                self.update_prob(selected_index, 0.2)
            elif choice == 'n':
                self.update_prob(selected_index, -0.2)
        selected_child.visit()

    def print(self, level = 0):
        print(f"{' ' * level}{self}")
        level += 1
        if not self.children.empty:
            for index, row in self.children.iterrows():
                row["children"].print(level)
        if not self.leaves.empty:
            for leaf in self.leaves:
                print(' ' * level + str(leaf))

if __name__ == "__main__":
    filename = Path('tree.pkl')
    root = pickle.load(open(filename, "rb")) if filename.exists() else DecisionTree("root")
    root.visit()
    print("Final tree:")
    root.print()
    pickle.dump(root, open(filename, "wb"))