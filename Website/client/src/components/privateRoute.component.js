import React from 'react';
import {Route,Redirect} from 'react-router-dom';
import PropTypes from 'prop-types';
import {connect} from 'react-redux';
const isEmpty = require("is-empty");

const PrivateRoute = ({component: Component,auth:Auth,...rest})=>(
    <Route
        {...rest}
        render = {
            props =>    
                !((localStorage.getItem("jwtToken")) === null) ? (<Component {...props}/>) : (<Redirect to='/signIn'/>)
        }
    />
)

// const mapStateToProps = state =>({
//         auth: state.auth,    
// });

// PrivateRoute.propTypes = {
//     auth: PropTypes.func.isRequired
// }

export default (PrivateRoute);