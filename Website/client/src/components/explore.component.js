import React,{Component} from 'react';
import axios from 'axios';
import LoginNavbar from './loggedInNavbar.component';
import '../index.css';
import PropTypes from 'prop-types';
import {connect} from 'react-redux';
import {Row,Col,Container} from 'react-bootstrap';
import Button from './button.component';
import {addStockToUserPortFolio,addStockToMarketWatch} from '../actions/userActions';

const buyBtn = {
    fontSize: "12px",
    borderRadius: "4px",
    color:"#4CAF50",
    backgroundColor:"white",
    border: "2px solid #4CAF50"
}

const addToMWBtn = {
    fontSize: "12px",
    borderRadius: "4px",
    color:"#f7a707",
    backgroundColor:"white",
    border: "2px solid #f7a707"
}

class Explore extends Component{
    
    constructor(){
        super();
        this.state = {
            keyword:"",
            matchedStocks : [],
            quantityOfStocks: []
        }   
        this.onBuyClicked = this.onBuyClicked.bind(this);
        this.onaddToMWClicked = this.onaddToMWClicked.bind(this);
        
    }

    onChange = e =>{  
        this.setState({[e.target.id]:e.target.value})
    }

    onSubmit = e =>{
        e.preventDefault()
        this.setState({
            keyword: e.target.value
        });
        const keyWord = this.state.keyword;
        const apiCall = "https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords="+keyWord+"&apikey=QAJSMECMPEBFVU2G"
        
        axios.get(apiCall)
            .then((res)=>{
                var Stocks = res.data.bestMatches;
                var len = Stocks.length;
                var stocks = Stocks.slice(0,Math.min(4,len))
                var len1 = stocks.length;
                var temp = [];
                for (var i=0;i<len1;i++){
                    temp.push(0);
                }
                console.log(temp);
                stocks.map(function(stock,i){
                    const apiCallForEachStock = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol="+stock['1. symbol']+"&apikey=QAJSMECMPEBFVU2G"
                    axios.get(apiCallForEachStock)
                        .then((res)=>{
                            stocks[i].info = (res.data["Global Quote"]);
                            this.setState ({
                                matchedStocks:stocks,
                                quantityOfStocks:temp
                            })
                        })   
                }.bind(this));
                
            })
    }

    onChangeEachStock = (e) =>{
        var temp = this.state.quantityOfStocks;
        temp[e.target.id] = e.target.value;
        this.setState({
            quantityOfStocks:temp
        })
    }

    onBuyClicked = (i,e) =>{
        e.preventDefault();
        const quantity = this.state.quantityOfStocks[i];
        const price = this.state.matchedStocks[i].info["05. price"];
        axios.get('http://localhost:5000/user/getUserInfo/'+localStorage.getItem("email"))
            .then((res)=>{
                const val = 555000 - res.data.totalInvestment;
                if(val >= (quantity*price)){
                    this.props.addStockToUserPortFolio(localStorage.getItem("email"),
                                                        quantity,
                                                        this.state.matchedStocks[i]["1. symbol"],
                                                        price
                                                        );
                }else{
                    alert("Not enough balance");
                }                                    
            })
    }

    onaddToMWClicked = (i,event) =>{
        event.preventDefault();
        axios.get('http://localhost:5000/user/getUserInfo/'+localStorage.getItem("email"))
            .then((res)=>{
                
                this.props.addStockToMarketWatch(localStorage.getItem("email"),
                                                this.state.matchedStocks[i]["1. symbol"]);
                                               
            })
    }

    render(){
        
        return (
            <div>
                <LoginNavbar name = "Saikumar Nalla"/>
                <div className ="wrap">
                    <div className="search">
                        <input 
                            onChange = {this.onChange}
                            value = {this.state.keyword}
                            type="text" 
                            id ="keyword"
                            className="searchTerm" 
                            placeholder="Search stocks using keywords"
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
                <div className = "matchingStocks">
                    {this.state.matchedStocks.length>0 ?
                     <div>
                        <h3 className="colText">The top 4 stocks as per your search are :</h3>
                            <Container>
                                <Row style = {{color:"#00B4CC"}} className ="rowStyle">
                                    <Col xs ={3}>
                                            SHARE SYMBOL
                                    </Col>
                                    <Col xs={5}>
                                            PRICE & CHANGE
                                    </Col>
                                    <Col xs={2}>
                                            CLICK TO BUY
                                    </Col>
                                    <Col xs={2}>
                                            ADD TO WATCH
                                    </Col>
                                </Row>
                                {
                                    this.state.matchedStocks.map((stock,i)=>{
                                        
                                        return(
                                        <div key ={i}>   
                                            <hr className="greyLine" />
                                            <Row className = "rowStyle" >
                                                        <Col xs ={3}>
                                                            {stock['1. symbol']}
                                                        </Col>
                                                        <Col xs ={3}>
                                                            {
                                                                !(stock.info)
                                                                ?
                                                                null
                                                                :
                                                                [
                                                                    !(stock.info['09. change'][0] === '-') ? 
                                                                        <div >
                                                                            <div>${stock.info['05. price']}</div>
                                                                            <div style={{display:"inline block",color:"green"}}>
                                                                                <i style ={{padding:"6px"}} className="fa fa-caret-up"></i>
                                                                                {stock.info['09. change']}%
                                                                            </div>
                                                                        </div>
                                                                        :
                                                                        <div >
                                                                           <div>${stock.info['05. price']}</div>
                                                                           <div style={{display:"inline block",color:"red"}}>
                                                                                <i style ={{padding:"6px"}} className="fa fa-caret-down"></i>
                                                                                {stock.info['09. change']}%
                                                                           </div>
                                                                        </div>
                                                                ]
                                                   
                                                            }
                                                        </Col>
                                                        <Col xs ={4}>
                                                            <form noValidate onSubmit = {this.onBuyClicked.bind(this,i)}>
                                                                <input
                                                                    type = "number"
                                                                    val={this.state.quantityOfStocks[parseInt(i)]}
                                                                    id = {i}
                                                                    onChange = {this.onChangeEachStock}
                                                                    placeholder = "No. of stocks"
                                                                />
                                                                <Button type="submit" style={buyBtn} label = "Buy"/>                                                            
                                                            </form>
                                                        </Col>
                                                        <Col xs ={2}>
                                                            <Button style={addToMWBtn} handleClick = {this.onaddToMWClicked.bind(this,i)} label = "Add to market watch"/>   
                                                        </Col>
                                                </Row>
                                            </div> 
                                        );
                                    })
                                }
                            </Container>
                        </div>
                        : <h3></h3> 
                    }   
                </div>
            </div>
        );
    }   
}

const mapStateToProps = state =>({
    auth: state.auth,
    user: state.user
}) 

Explore.propTypes = {
    auth:PropTypes.object.isRequired,
    user:PropTypes.object.isRequired
}

export default connect(mapStateToProps,{addStockToUserPortFolio,addStockToMarketWatch})(Explore);