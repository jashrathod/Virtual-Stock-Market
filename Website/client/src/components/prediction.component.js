import React,{Component}  from 'react';
import '../index.css';
import LoginNavbar from './loggedInNavbar.component';
import axios from 'axios';
import {Row,Col,Container} from 'react-bootstrap';

class Prediction extends Component{
    
    constructor(){
        super();
        this.state = {
            symbolEntered:"",
            predictedPrice : 0
        }
    }

    onChange=(e)=>{
        this.setState({
            [e.target.id]:e.target.value
        })
    }

    onSubmit = (e) =>{
        e.preventDefault();
        this.setState({
            symbolEntered:e.target.value
        })
        var symbol = this.state.symbolEntered;
        console.log(this.state.symbolEntered);
        const callToServer = "http://localhost:5000/user/getPrediction/"+this.state.symbolEntered;
        axios.get(callToServer)
             .then((res)=>{
                 this.setState({
                       symbolEntered:symbol,
                       predictedPrice: res.data[0] 
                 })
             })
    }
    
    render(){
        return(
            <div>
                <LoginNavbar name = "Saikumar Nalla"/>
                <div className ="wrap">
                    <div className="search">
                        <input 
                            onChange = {this.onChange}
                            value = {this.state.keyword}
                            type="text" 
                            id ="symbolEntered"
                            className="searchTerm" 
                            placeholder="Enter the stock for which you want to predict its price"
                        />
                        <button 
                            type="submit" 
                            className="searchButton"
                            onClick = {this.onSubmit}
                        >
                            <i className="fa fa-search"></i>
                        </button>
                    </div>
                </div>
                {
                    this.state.predictedPrice != 0 
                    ?
                    <Container style ={{margin:"auto"}}>
                        <Row className="rowStyle">
                            <Col className = "colText" xs={5}>
                                Stock Entered
                            </Col>
                            <Col className = "colText" xs={5}>
                                Predicted Price
                            </Col>
                            <Col className = "colText" xs={5}>
                                {this.state.symbolEntered}
                            </Col>
                            <Col className = "colText" xs={5}>
                                ${this.state.predictedPrice}
                            </Col>
                        </Row>
                        
                    </Container>
                    :
                    <h3></h3>
                }
            </div>    
        )
    }
}

export default Prediction;