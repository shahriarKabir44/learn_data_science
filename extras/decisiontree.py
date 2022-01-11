from a import classify, info_gain


data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]


def unique_vals(data, col):
    return set([n[col] for n in data])


def class_counts(data):
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    def __init__(self, col, val) -> None:
        self.col = col
        self.val = val

    def __repr__(self) -> str:
        cond = '=='
        if isinstance(self.val, int) or isinstance(self.val, float):
            cond = '>='
        return 'is {} {} {}?'.format(header[self.col], cond, str(self.val))

    def match(self, data) -> bool:
        vl = data[self.col]
        if isinstance(vl, int) or isinstance(vl, float):
            return vl >= self.val
        else:
            return vl == self.val


def partition(data, question) -> tuple:
    tru = []
    fal = []
    for n in data:
        if question.match(n):
            tru.append(n)
        else:
            fal.append(n)
    return (tru, fal)


def gini(data):
    vis = {}
    for n in data:
        if n[-1] not in vis:
            vis[n[-1]] = 0
        vis[n[-1]] += 1
    impur = 1
    for n in vis:
        impur -= (vis[n]/len(data))**2
    return impur


def gain(tru, fls, cur):
    p = (len(tru)/(len(tru)+len(fls)))
    return cur-p*(gini(tru))-(1-p)*gini(fls)


def bestSplit(data):
    feaures = len(data[0])-1
    gain = 0
    uncertainity = gini(data)
    bestq = None

    for n in range(feaures):
        tem = set([row[n] for row in data])
        tru = []
        fls = []
        for k in tem:

            ques = Question(n, k)

            tru, fls = partition(data, ques)
        if len(tru)*len(fls) == 0:
            continue
        gan = info_gain(tru, fls, uncertainity)
        if gan > gain:
            gain = gan
            bestq = ques
    return (gain, bestq)


class Leaf:
    def __init__(self, data) -> None:
        self.preds = class_counts(data)


class Node:
    def __init__(self, ques, tru, fls) -> None:
        self.ques = ques
        self.tru = tru
        self.fls = fls


def train(data):
    gain, ques = bestSplit(data)

    if gain == 0:
        return Leaf(data)

    tru, fls = partition(data, ques)

    trubran = train(tru)
    falbran = train(fls)

    return Node(ques, trubran, falbran)


def dfs(data, node):
    if isinstance(node, Leaf):
        return node.preds

    elif isinstance(node, Node):
        if node.ques.match(data):
            return dfs(data, node.tru)
        else:
            return dfs(data, node.fls)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.preds)
        return

    print(spacing + str(node.ques))

    print(spacing + '--> True:')
    print_tree(node.tru, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.fls, spacing + "  ")


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


tree = train(data)
print_tree(tree)


test = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

res = dfs(test[2], tree)
print(print_leaf(res))
