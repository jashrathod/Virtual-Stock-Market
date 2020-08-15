import React,{Component} from 'react';
import LoginNavbar from './loggedInNavbar.component';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import PropTypes from 'prop-types';
import {getUserInfo,sellStock} from '../actions/userActions';
import {connect} from 'react-redux';
import '../index.css'
import Button from './button.component';
import axios from 'axios';
import {logOutUser} from '../actions/authActions';

var Email="";

const SellBtn = {
    fontSize: "12px",
    borderRadius: "4px",
    color:"#f7a707",
    backgroundColor:"white",
    border: "2px solid #f7a707"
}

class MainPage extends Component{

    constructor(){
        super();
        this.state = {
            userName : "",
            marginAvailable : 0,
            totalInvestment : 0,
            value : 0,
            userStocks: [],
            userMarketWatch:[]
        }
    }

    componentWillReceiveProps(nextProps){

        if(nextProps.user.userInfo){    
            var stocks = nextProps.user.userInfo.stocks;
            var value = 0;
            console.log(stocks);
            stocks.map((stock,i)=>{
                const apiCallForEachStock = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol="+stock.symbol+"&apikey=QAJSMECMPEBFVU2G";
                axios.get(apiCallForEachStock)
                .then((res) =>{
                    stocks[i].info = res.data["Global Quote"]
                    stocks[i].profitAndLoss = (parseFloat(stocks[i].info["05. price"])-parseFloat(stocks[i].price))/100.0
                    value = value+stocks[i].quantity*parseFloat(stocks[i].info["05. price"])
                    this.setState({
                        userName: nextProps.user.userInfo.name,
                        marginAvailable: (555000-nextProps.user.userInfo.totalInvestment).toFixed(2),
                        totalInvestment: (nextProps.user.userInfo.totalInvestment).toFixed(2),
                        value: (555000-nextProps.user.userInfo.totalInvestment + value).toFixed(2),
                        userStocks: stocks,
                        userMarketWatch: nextProps.user.userInfo.marketWatch
                    })
                })
            })

            var marketWatchList = nextProps.user.userInfo.marketWatch
            marketWatchList.map((stock,i)=>{
                const apiCallForEachStock = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol="+stock.symbol+"&apikey=QAJSMECMPEBFVU2G";
                axios.get(apiCallForEachStock)
                .then((res) =>{
                    marketWatchList[i].info = res.data["Global Quote"]
                    this.setState({
                        userName: nextProps.user.userInfo.name,
                        marginAvailable: (555000-nextProps.user.userInfo.totalInvestment).toFixed(2),
                        totalInvestment: (nextProps.user.userInfo.totalInvestment).toFixed(2),
                        value: (555000-nextProps.user.userInfo.totalInvestment + value).toFixed(2),
                        userStocks: stocks,
                        userMarketWatch: marketWatchList
                    })
                })
            })
            this.setState({
                userName: nextProps.user.userInfo.name,
                marginAvailable: (555000-nextProps.user.userInfo.totalInvestment).toFixed(2),
                totalInvestment: (nextProps.user.userInfo.totalInvestment).toFixed(2),
                value: (555000-nextProps.user.userInfo.totalInvestment + value).toFixed(2),
                userStocks: stocks,
                userMarketWatch: marketWatchList
            })
        }else{
            this.props.history.push('/signUp');
        }
    }

    componentDidMount(){
        if(localStorage.email){
            Email = localStorage.email;
        }
        this.props.getUserInfo(Email)
    }

    onClickSell(i,e){
        this.props.sellStock(Email,i);
    }

    onClickLogOut=(e)=>{
        this.props.logOutUser();
    }

    render(){
        return (
            <div>
                <LoginNavbar name = {this.state.userName} handleClick = {this.onClickLogOut}/>
                <Container>
                    <Row>
                        <Col style={{textAlign:"left"}} xs ={12} md={4}>
                            <p style={{color:"grey",textAlign:"left"}} className="numbers">Market Watch</p>
                            <br/>
                             {!this.state.userMarketWatch.length > 0
                                ? 
                                <p style={{margin:0,padding:0}}>No stocks added to market watch</p>
                                :
                                <Container>
                                {
                                    this.state.userMarketWatch.map((stock)=>{
                                        return(
                                            <div>
                                                <hr className = "greyLine"/>
                                                <Row>
                                                <Col xs = {4}>
                                                        {stock.symbol}
                                                </Col>
                                                <Col xs ={6}>
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
                                            </Row>
                                            </div>        
                                        )
                                    })
                                }
                            </Container>   
                              } 
                        </Col>
                        <Col xs={12} md={8}>
                            <Container>
                                <Row xs = {4}>
                                    <Col>
                                        <Container>
                                            <Row>
                                                <Col style={{color:"grey",textAlign:"left"}} className = "colText" xs={12}>
                                                    <p className = "numbers">Hi, {this.state.userName}</p>
                                                </Col>
                                            </Row>
                                            <hr className = "greyLine"/>
                                            <Container>
                                                <Row>
                                                    <Col className = "colText" xs={5}>
                                                        <div>
                                                            <p style={{color:"grey"}} className = "numbers">{this.state.marginAvailable}</p>
                                                            <p style={{color:"grey"}} className ="info">Margin Available</p>
                                                        </div>
                                                    </Col>
                                                    <Col className = "colText" xs={5}>
                                                        <div>
                                                            <p style={{color:"#0099ff"}} className = "numbers">{this.state.totalInvestment}</p>
                                                            <p style={{color:"grey"}} className ="info">Margin Used</p>
                                                        </div>
                                                    </Col>
                                                </Row>
                                                <Row>
                                                    <Col className = "colText">
                                                        <div>
                                                            <p style={{color:"#0099ff"}} className = "numbers">{this.state.value}</p>
                                                            <p style={{color:"grey"}} className ="info">Value</p>
                                                        </div>
                                                    </Col>
                                                </Row>
                                            </Container>
                                        </Container>
                                    </Col>
                                </Row>
                                <Row xs = {8}>
                                    <Col>
                                        <hr className="greyLine"/>
                                        <br/>
                                        <p style={{color:"grey",textAign:"left"}} className = "numbers">Your Stock Holdings: </p> 
                                        <br/>
                                        <Container>
                                            
                                            {
                                                this.state.userStocks.length > 0 
                                                ? 
                                                <div>
                                                    <Row style = {{color:"#00B4CC"}} className ="rowStyle">
                                                        <Col xs ={3}>
                                                                SHARE SYMBOL
                                                        </Col>
                                                        <Col xs={3}>
                                                                PRICE & CHANGE
                                                        </Col>
                                                        <Col xs={2}>
                                                                QUANTITY
                                                        </Col>
                                                        <Col xs={2}>
                                                                CLICK TO SELL
                                                        </Col>
                                                        <Col xs={2}>
                                                                P&L
                                                        </Col>
                                                    </Row>
                                                    {
                                                        this.state.userStocks.map((stock,i) =>{
                                                            return(
                                                                <div key ={i}>   
                                                           
                                                            <hr className="greyLine" />
                                                            <Row className = "rowStyle" >
                                                                        <Col xs ={3}>
                                                                            {stock.symbol}
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
                                                                        <Col xs = {2}>
                                                                                {stock.quantity}
                                                                        </Col>
                                                                        <Col xs ={2}>
                                                                                <Button type="submit" handleClick = {this.onClickSell.bind(this,i)} style={SellBtn} label = "Sell"/>                                                           
                                                                        </Col>
                                                                        <Col xs = {2}>
                                                                            {
                                                                                stock.profitAndLoss >= 0 
                                                                                ? 
                                                                                <p style={{color:"green"}}>{stock.profitAndLoss}%</p>
                                                                                :
                                                                                <p style ={{color:"red"}} >{stock.profitAndLoss}%</p>
                                                                            }
                                                                        </Col>
                                                                </Row>
                                                        </div>
                                                            )
                                                        })
                                                    }    
                                                </div>
                                            
                                                :
                                                <div style ={{textAlign:"center"}}>
                                                    <p style={{margin:0,padding:0}}>Currently you don't own any stocks</p>
                                                    <p style={{margin:0,padding:0}}>Go to the explore section to invest in some stocks</p>    
                                                </div>
                                            }
                                        </Container>
                                    </Col>
                                </Row>
                            </Container>
                        </Col>    
                    </Row>
                </Container>
            </div>
        );
    }
}

MainPage.propTypes = {
    getUserInfo: PropTypes.func.isRequired,
    sellStock: PropTypes.func.isRequired,
    logOutUser: PropTypes.func.isRequired,
    user: PropTypes.object.isRequired,
};

const mapStateToProps = state =>({
    auth: state.auth,
    user: state.user,
});

export default connect(mapStateToProps,{getUserInfo,sellStock,logOutUser})(MainPage);