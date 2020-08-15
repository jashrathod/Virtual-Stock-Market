var {PythonShell} = require('python-shell');
    var symbolEntered = "TCS"
    var options = {
        mode: 'text',
        scriptPath:'/home/171070030/virtualStockMarket/backend/routes',
        args: [symbolEntered]
    };

    PythonShell.run("predictions.py", options, function (err, results) {
        if (err) throw err;
        console.log(results);
        
    });