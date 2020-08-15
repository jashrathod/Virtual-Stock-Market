import axios from 'axios';
import {GET_ERRORS,SET_USER_INFO} from './types';

export const getUserInfo = (email) => dispatch =>{
    axios.get('http://localhost:5000/user/getUserInfo/'+email)
         .then(res =>{
             const info = res.data;
             dispatch(setUserInfo(info));  
         })
         .catch(err => {
                dispatch({
                    type: GET_ERRORS,
                    payload: err
                })
         });
} 

export const setUserInfo = (info) =>{
    return {
        type:SET_USER_INFO,
        payload:info
    }
}

export const addStockToUserPortFolio = (email,quantity,stockSymbol,priceBoughtAt) => dispatch =>{
    const stockToBeAdded = {
        symbol:stockSymbol,
        quantity: quantity,
        price: priceBoughtAt
    }
    const data = {
        email: email,
        stock: stockToBeAdded
    }
    axios.put('http://localhost:5000/user/updateUserPortfolio',data)
         .then((res)=>{
             alert("Stock bought successfully")
            dispatch(setUserInfo(res.data))
         })
}

 export const addStockToMarketWatch = (email,stockSymbol) => dispatch =>{
     const data  = {
         email:email,
         symbol: stockSymbol
     }
     axios.put('http://localhost:5000/user/updateUserMarketWatch',data)
          .then((res)=>{
                alert("Stock added to market watch successfully")
                dispatch(setUserInfo(res.data)) 
          })
 }

 export const sellStock = (email,index) => dispatch =>{
    const data = {
        email: email,
        index: index
    } 
    axios.put('http://localhost:5000/user/sellStock',data)
         .then((res) =>{
             alert("Stock sold successfully");
             dispatch(setUserInfo(res.data));
         })
 }
 