import axios from 'axios';
import jwt_decode from 'jwt-decode';
import setAuthToken from '../utils/setAuthToken';


import {SET_CURRENT_USER,USER_LOADING,GET_ERRORS} from './types';

export const registerUser = (userData,history) => dispatch =>{
    axios.post('http://localhost:5000/user/signUp',userData)
         .then(res => history.push('/signIn'))
         .catch(err => {
            dispatch({
                type: GET_ERRORS,
                payload: err
            })
         });
}

export const loginUser = (userData) => dispatch =>{
    axios.post('http://localhost:5000/user/signIn', userData)
         .then(res =>{
             const {token} = res.data;
             localStorage.setItem("email",userData.email)
             localStorage.setItem("jwtToken",token);
             setAuthToken(token);
             const decoded = jwt_decode(token);
             dispatch(setCurrentUser(decoded));
         })
         .catch(err =>{
             dispatch({
                 type:GET_ERRORS,
                 payload:err
             })
         })
} 


export const setCurrentUser = (decoded) => {
    return  {
        type: SET_CURRENT_USER,
        payload: decoded
    };
}

export const setUserLoading = () =>{
    return {
        type: USER_LOADING
    };
}

export const logOutUser = () => dispatch =>{
    localStorage.removeItem("jwtToken");
    localStorage.removeItem("email");
    setAuthToken(false);
    dispatch(setCurrentUser({}));
}