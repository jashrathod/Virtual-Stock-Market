import {SET_USER_INFO} from '../actions/types';

const initialState = {
    userInfo:{}
};

export default function(state = initialState,action){
    switch(action.type){
        case SET_USER_INFO:
            return{
                userInfo: action.payload,
            }
        default:
            return state;    
    }
}