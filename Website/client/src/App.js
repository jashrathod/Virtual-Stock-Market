import React from 'react';
import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter as Router, Route,Switch} from "react-router-dom";
import LandingPage from "./components/landingPage.component";
import SignUp from './components/signUp.component';
import SignIn from './components/signIn.component';
import MainPage from './components/mainpage.component';
import Explore from './components/explore.component';
import jwt_decode from 'jwt-decode';
import {setCurrentUser,logOutUser} from './actions/authActions';
import setAuthToken from './utils/setAuthToken';
import PrivateRoute from './components/privateRoute.component';
import {Provider} from 'react-redux';
import store from './store';
import {connect} from 'react-redux'
import Prediction from './components/prediction.component';
import PropTypes from 'prop-types'

if(localStorage.token){
  const jwtToken = localStorage.token;
  setAuthToken(jwtToken);

  const decoded = jwt_decode(jwtToken);
  store.dispatch(setCurrentUser(decoded));


  const currentTime = Date.now();
  if(decoded.exp < currentTime){
    store.dispatch(logOutUser())
    window.location.href = './login'
  }
}

const App = (props) => {
  return (
   
      <Router>
        <div>
            
            <Route exact path='/'  component = {LandingPage}/>
            <Route path='/signUp' component = {SignUp} />
            <Route path='/signIn' component = {SignIn} />
            <Switch>
                <PrivateRoute exact path = '/dashboard' auth = {props.auth} component = {MainPage} />
                <PrivateRoute exact path = '/explore' auth = {props.auth} component = {Explore} />
                <PrivateRoute exact path = '/predict' auth = {props.auth} component = {Prediction} />
            </Switch>
        </div>
      </Router>
    
  );
}

App.propTypes = {
  auth: PropTypes.object.isRequired,
}

const mapStateToProps = state => ({
  auth:state.auth
});

export default connect(mapStateToProps)(App);
