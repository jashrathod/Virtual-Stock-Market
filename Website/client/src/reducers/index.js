import {combineReducers} from 'redux';
import authReducer from './authReducers';
import userReducer from './userReducer'; 
import errorReducer from './errorReducers';

export default combineReducers({
    auth: authReducer,
    user: userReducer,
    errors:errorReducer
});
