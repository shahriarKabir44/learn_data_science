var data = [
    ['sunny', 'hot', 'high', 'false', 'no'],
    ['sunny', 'hot', 'high', 'true', 'no'],
    ['overcast', 'hot', 'high', 'false', 'yes'],
    ['rainy', 'mild', 'high', 'false', 'yes'],
    ['rainy', 'cool', 'normal', 'true', 'no'],
    ['overcast', 'cool', 'normal', 'true', 'yes'],
    ['sunny', 'mild', 'high', 'false', 'no'],
    ['sunny', 'cool', 'normal', 'false', 'yes'],
    ['rainy', 'mild', 'normal', 'false', 'yes'],
    ['sunny', 'mild', 'normal', 'true', 'yes'],
    ['overcast', 'mild', 'high', 'true', 'yes'],

    ['rainy', 'cool', 'normal', 'false', 'yes'],
    ['overcast', 'hot', 'normal', 'false', 'yes']

]

var test = [
    ['rainy', 'mild', 'high', 'true', 'no'],

]

var header = ['outlook', 'temp', 'humidity', 'windy', 'play']

class question {
    constructor(col, vl) {
        this.col = col
        this.vl = vl
    }
    match(data) {
        var val = data[this.col]
        if (isNaN(val)) {
            return val == this.vl
        }
        else return val >= this.vl
    }
    toString() {
        var condition = '=='
        if (!isNaN(this.vl)) condition = '>='
        return `is ${header[this.col]} ${condition} ${this.vl}?`
    }
}

function uniq(data, col) {
    var st = new Set()
    for (let n = 0; n < data.length; n++) {
        st.add(data[n][col])
    }
    var dat = []
    st.forEach(dt => {
        dat.push(dt)
    })
    return dat;
}

function classcnt(data) {
    var cnt = {}
    for (let n = 0; n < data.length; n++) {
        var lb = data[n].slice(-1)[0]
        if (cnt[lb] == null) cnt[lb] = 0;
        cnt[lb]++;
    }
    return cnt
}

function gini(data) {
    var vis = {}
    for (let n = 0; n < data.length; n++) {
        var y = data[n].slice(-1)[0];
        if (vis[y] == null) vis[y] = 0

        vis[y]++;
    }
    var imp = 1
    for (let n in vis) {
        imp -= (vis[n] / data.length) ** 2
    }
    return imp
}

function gain(tru, fls, cur) {
    var p = (tru.length) / (tru.length + fls.length)
    return cur - p * gini(tru) - (1 - p) * gini(fls)
}

function partition(data, ques) {
    var trus = []
    var fals = []
    for (let n in data) {
        if (ques.match(data[n])) trus.push(data[n])
        else fals.push(data[n])
    }
    return { trus: trus, fals: fals }
}

function bestSplit(data) {
    var bestq = null
    var gainx = 0
    var uncert = gini(data)
    var feats = data[0].length - 1;
    for (let colnm = 0; colnm < feats; colnm++) {
        var tem = new Set()
        for (let n = 0; n < data.length; n++) {
            tem.add(data[n][colnm])
        }
        tem = Array.from(tem)
        tem.forEach(row => {
            var qs = new question(colnm, row)

            var { trus, fals } = partition(data, qs)
            var gan = gain(trus, fals, uncert)
            if (gan >= gainx) {
                gainx = gan;
                bestq = qs
            }
        })
    }
    return { gainx: gainx, bestq: bestq }
}

class Leaf {
    constructor(data) {
        this.preds = classcnt(data)
    }
}
class Node {
    constructor(tru, fals, ques) {
        this.tru = tru
        this.fals = fals
        this.ques = ques
    }
}

function train(data) {
    var { gainx, bestq } = bestSplit(data)
    if (!gainx) return new Leaf(data)

    var { trus, fals } = partition(data, bestq);
    var trubran = train(trus)
    var falsbran = train(fals)
    return new Node(trubran, falsbran, bestq)
}

function dfs(data, node) {
    if (node.constructor.name == 'Leaf') {
        return node.preds;
    }
    else {
        if (node.ques.match(data)) {
            return dfs(data, node.tru)
        }
        else return dfs(data, node.fals)
    }
}




var tree = train(data)

var pred = dfs(test[0], tree)

console.log(pred)