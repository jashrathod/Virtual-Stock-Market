const router = require('express').Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const validateRegisterInput = require('../validation/register');
const validateLoginInput = require('../validation/login');
const User = require('../models/User');

router.route('/').get((req,res)=>{
   
});

router.route('/getUserInfo/:email').get((req,res)=>{
    const email = req.params.email;
    User.findOne({email})
        .then((user) =>{
            return res.json(user);
        })
});

router.route('/signUp').post((req,res)=>{
   const {errors,isValid} = validateRegisterInput(req.body);
   
   if(!isValid){
       return  res.status(400).json(errors);
   } 

   User.findOne({email : req.body.email})
       .then((user) => {
                
                if(user){
                    return res.status(400).json({email: "Email already exists."});       
                }else{
                    const newUser = new User({
                        name : req.body.name,
                        email: req.body.email,
                        password: req.body.password
                    });
                    bcrypt.genSalt(10,(err,salt)=>{
                        bcrypt.hash(newUser.password,salt,(err,hash)=>{
                            if(err)throw err;
                            newUser.password = hash;
                            newUser.save()
                                   .then(user => {return res.json(user)})
                                   .catch(err => {return res.status(400).json(err)});
                        });
                    });
                }

            }
       )
});

router.route('/signIn').post((req,res)=>{
    
    const {errors,isValid} = validateLoginInput(req.body);
    if(!isValid){
        return  res.status(400).json(errors);      
    }

    const email = req.body.email;
    const password = req.body.password;

    User.findOne({email})
        .then(
            (user)=>{
                if(!user){
                    res.status(400).json({emailnotfound : "Email not found"});
                }else{
                    bcrypt.compare(password,user.password)
                          .then(
                              (isMatch) =>{
                                  if(isMatch){
                                        const payload = {
                                            id: user.id,
                                            name: user.name
                                        };
                                        jwt.sign(
                                            payload,
                                            process.env.secretOrKey,
                                            {
                                                expiresIn : 31556926
                                            },
                                            (err,token)=>{
                                                   res.json({
                                                       success: true,
                                                       token: "Bearer "+token
                                                   }) ;
                                            }                                            
                                        );
                                  }else{
                                      return  res.status(400).json({password: "Password entered is not correct"});
                                  }
                              }
                          );
                }
            }
        )

});

router.route('/updateUserPortfolio').put((req,res)=>{
    const newStockToAdd = req.body.stock;
    const email = req.body.email;
    User.findOne({email})
        .then((user) =>{
            user.stocks.push(
                newStockToAdd
            );
            user.totalInvestment = user.totalInvestment + (newStockToAdd.quantity*newStockToAdd.price);
            user.save()
                .then((updatedUser) =>{
                    return res.json(updatedUser);
                });
        });
});

router.route('/updateUserMarketWatch').put((req,res)=>{
    const stockSym = {
        symbol:req.body.symbol
    };    
    const email = req.body.email
    User.findOne({email})
        .then((user) =>{
            user.marketWatch.push(stockSym)
            user.save()
                .then((updatedUser) =>{
                    return res.json(updatedUser);
                });
        })
})


router.route('/sellStock').put((req,res)=>{
    const email = req.body.email;
    const index = req.body.index;
    User.findOne({email})
        .then((user) =>{
            user.totalInvestment = user.totalInvestment - (user.stocks[index].quantity*user.stocks[index].price)
            user.stocks.splice(index,1)
           
            user.save()
                .then((updatedUser) =>{
                    console.log(updatedUser);
                    res.json(updatedUser);
                });
        })
})

router.route('/getPrediction/:symbol').get((req,res)=>{
    const symbolEntered = req.params.symbol;
    console.log(symbolEntered);
    var {PythonShell} = require('python-shell');

    var options = {
        mode: 'text',
        scriptPath:'/home/171070030/virtualStockMarket/backend/routes',
        args: [symbolEntered]
    };

    PythonShell.run("predictions.py", options, function (err, results) {
        if (err) throw err;
        console.log(results);
        return res.json(results);
    });
    
})

module.exports = router;