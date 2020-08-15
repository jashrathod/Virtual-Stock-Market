import React,{Component} from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import '../index.css';
import Button from './button.component';
import { connect } from 'react-redux';
import {loginUser} from '../actions/authActions';
import classnames from 'classnames'; 
import PropTypes from 'prop-types';

const btnStyle = {
    width:'40%',
    position:'relative',
    left:'4%'
}

class SignIn extends Component{

    constructor(){
        super();
        this.state ={
            email: "",
            password: "",
            errors:{}
        };
    }

    componentDidMount(){
        if(this.props.auth.isAuthenticated){
            this.props.history.push('/dashboard');
        }
    }

    componentWillReceiveProps(nextProps){
        if(nextProps.auth.isAuthenticated){
            this.props.history.push('/dashboard')
        }
        if(nextProps.errors){
            this.setState({
                errors:nextProps.errors
            })
        }
    }

    onChange = e =>{
        
        this.setState({[e.target.id]:e.target.value})
    };

    onSubmit = e =>{
        e.preventDefault();

        const userLoggingIn = {
            email: e.target.email.value,
            password: e.target.password.value,
        };
        this.props.loginUser(userLoggingIn);
        
    }

    render(){
        const {errors} = this.state;
        return(
            <Container>
                <Row className = "justify-content-md-center">              
                    <Col xs={3}>
                    </Col>                 
                    <Col xs={6}>
                        <div className = "sign-up">
                            <h1>VSM</h1>
                            <form noValidate onSubmit = {this.onSubmit}>
                                
                                <div className="inputFields">
                                    <label htmlFor="email">Email</label>
                                    <br/>
                                    <input
                                        onChange = {this.onChange}
                                        value = {this.state.email}
                                        error={errors.email}
                                        id = "email"
                                        type = "email"
                                        className={classnames("", {
                                            invalid: errors.email || errors.emailnotfound
                                        })}
                                    />
                                    <span className="red-text">
                                        {errors.email}
                                        {errors.emailnotfound}
                                    </span>
                                </div>

                                <div className="inputFields">
                                    <label htmlFor="password">Password</label>
                                    <br/>
                                    <input
                                        onChange = {this.onChange}
                                        value = {this.state.password}
                                        error={errors.password}
                                        id = "password"
                                        type = "password"
                                        className={classnames("", {
                                            invalid: errors.password || errors.passwordincorrect
                                        })}
                                    />
                                    <span className="red-text">
                                        {errors.password}
                                        {errors.passwordincorrect}
                                    </span>
                                </div>
                                <div>
                                    <Button style = {btnStyle} label = "LOGIN" type="submit"/>
                                </div>
                                <p> New User? <a href = "/signUp"> Register</a></p>
                            </form>
                        </div>
                    </Col>
                    <Col xs={3}>
                    </Col>
                </Row>
            </Container>
            
        )
    }
}

const mapStateToProps = state =>({
    auth: state.auth,
    errors: state.errors
})

SignIn.propTypes = {
    loginUser :PropTypes.func.isRequired,
    auth :PropTypes.object.isRequired,
    errors:PropTypes.object.isRequired
}

export default connect(mapStateToProps,{loginUser})(SignIn)