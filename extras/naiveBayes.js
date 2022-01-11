var data = [
    ['sunny', 'hot', 'high', 'false', 'no'],
    ['sunny', 'hot', 'high', 'true', 'no'],
    ['overcast', 'hot', 'high', 'false', 'yes'],
    ['rainy', 'mild', 'high', 'false', 'yes'],
    ['rainy', 'cool', 'normal', 'true', 'no'],
    ['overcast', 'cool', 'normal', 'true', 'yes'],
    ['sunny', 'mild', 'high', 'false', 'no'],
    ['rainy', 'mild', 'normal', 'false', 'yes'],
    ['sunny', 'mild', 'normal', 'true', 'yes'],
    ['overcast', 'mild', 'high', 'true', 'yes'],
    ['rainy', 'mild', 'high', 'true', 'no'],
    ['rainy', 'cool', 'normal', 'false', 'yes'],
    ['overcast', 'hot', 'normal', 'false', 'yes'],
    ['sunny', 'cool', 'normal', 'false', 'yes'],
]

var test = ['overcast', 'hot', 'normal', 'false', 'yes']


var header = ['outlook', 'temp', 'humidity', 'windy', 'play']

var ocrs = {}
var yes = 0
var no = 0
data.forEach(dt => {
    for (let n = 0; n < dt.length - 1; n++) {
        if (ocrs[dt[n]] == null) ocrs[dt[n]] = [0, 0, 0, 0]
        if (dt.slice(-1)[0] == 'no') {
            no++;
            ocrs[dt[n]][0]++;
        }
        else {
            yes++;
            ocrs[dt[n]][1]++;
        }
    }
})

for (let n in ocrs) {
    ocrs[n][2] = ocrs[n][0] / no
    ocrs[n][3] = ocrs[n][1] / yes
}

var a = 1
var b = 1

test.slice(0, test.length - 1).forEach(feat => {
    a *= ocrs[feat][2]
    b *= ocrs[feat][3]
})
var x = a / (a + b)
var y = b / (a + b)
var res = (x > y) ? 'no' : 'yes'
console.log(res)