import React,{Component} from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import '../index.css';
import {connect} from 'react-redux';
import Button from './button.component';
import PropTypes from 'prop-types';
import {registerUser} from '../actions/authActions';
import classnames from 'classnames'; 
import {withRouter} from 'react-router-dom';

const btnStyle = {
    width:'40%',
    position:'relative',
    left:'4%'
}

class SignUp extends Component{
    
    componentDidMount(){
        if(this.props.auth.isAuthenticated){
            this.props.history.push('/dashboard');
        }
    }

    componentWillReceiveProps(nextProps){
        if(nextProps.errors){
            this.setState({
                errors: nextProps.errors
            })
        }
    }

    constructor(){
        super();
        this.state = {
            name: "",
            email: "",
            password: "",
            password2: "",
            errors: {}
        };
    }

    onChange = e =>{
        this.setState({[e.target.id]:e.target.value})
    }

    onSubmit = e =>{
        e.preventDefault();
        const newUser = {
            name:this.state.name,
            email:this.state.email,
            password:this.state.password,
            password2:this.state.password2
        };
        
        this.props.registerUser(newUser,this.props.history);
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
                                    <label htmlFor="name">Name</label>
                                    <br/>
                                    <input
                                        onChange = {this.onChange}
                                        value ={this.state.name}
                                        error={errors.name}
                                        id = "name"
                                        type = "text"
                                        className = {classnames("",{
                                            invalid: errors.name
                                        })}
                                    />
                                    <span className = "red-text">{errors.name}</span>
                                </div>
                                <div className="inputFields">
                                    <label htmlFor="email">Email</label>
                                    <br/>
                                    <input
                                        onChange = {this.onChange}
                                        value ={this.state.email}
                                        error={errors.email}
                                        id = "email"
                                        type = "email"
                                        className = {classnames("",{
                                            invalid: errors.email
                                        })}
                                    />
                                    <span className = "red-text">{errors.email}</span>
                                </div>
                                <div className="inputFields">
                                    <label htmlFor="password">Password</label>
                                    <br/>
                                    <input
                                        onChange = {this.onChange}
                                        value ={this.state.password}
                                        error={errors.password}
                                        id = "password"
                                        type = "password"
                                        className = {classnames("",{
                                            invalid: errors.password
                                        })}
                                    />
                                    <span className = "red-text">{errors.password}</span>
                                </div>
                                <div className="inputFields">
                                    <label htmlFor="password2">Confirm Password</label>
                                    <br/>
                                    <input
                                        onChange = {this.onChange}
                                        value ={this.state.password2}
                                        error={errors.password2}
                                        id = "password2"
                                        type = "password"
                                        className = {classnames("",{
                                            invalid: errors.password2
                                        })}
                                    />
                                    <span className = "red-text">{errors.password2}</span>
                                </div>
                                <div>
                                <Button style = {btnStyle} label = "SIGN UP" type="submit"/>
                                </div>
                                <p>Already have an account?<a href = "/signIn">Login</a></p>
                            </form>
                        </div>
                    </Col>
               
                    <Col xs={3}>
                    </Col>
                </Row>
            </Container>
        );
    }
}

SignUp.propTypes = {
    registerUser : PropTypes.func.isRequired,
    auth: PropTypes.object.isRequired,
    errors: PropTypes.object.isRequired
}

const mapStateToProps = state =>({
    auth: state.auth,
    errors: state.errors
});


export default connect(mapStateToProps,{registerUser})(withRouter(SignUp));