import React,{Component} from 'react';
import '../index.css';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from './button.component';
import Navbar from './navbar.component';

var ReactRotatingText = require('react-rotating-text');


export default class LandingPage extends Component{
    render(){
        return(
            <div>
                    <Navbar/>
                    <div className = "bg">
                        <div className="bg-text">
                            <h1>Your global trading resource built</h1>
                            <h2>
                                <ReactRotatingText items={['to empower.','around your needs.','for beginnners.','to educate.','for new traders.']} color='Aqua'/>
                            </h2>
                            <div className = "btn-container">
                                <a href = "/signUp"><Button label = "Start for free"/></a>
                                <a href = "https://www.investopedia.com/articles/basics/06/invest1000.asp"><Button label = "Learn Basics of Stock market"/></a>
                            </div>
                        </div>
                    </div>
                    <div id = "howItWorks">
                        <Container>
                            <Row>   
                                <Col className = "colText" xs={12}>
                                <h3>Stock Investing as a life skill</h3>
                                <h4>Whether you are an explorer, a learner or a pro, you can get started immediately.</h4>
                                <hr className="blueLine"/>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs={12} md={6}>
                                    <p className = "number">01</p>
                                    <h2>EXPLORERS</h2>
                                    <p>First time in the world of investing?VSM is here to guide complete beginners
                                    (we use the term Explorers) through the stock markets and kick-off the learning
                                    process.The simple interface and helpful content will ease you into the world of
                                    trading.It's easier than it looks.Go ahead and try it.
                                    </p>
                                </Col>
                                <Col xs ={0} md={6}>
                                </Col>
                            </Row>
                            <hr/>
                            <Row>
                                <Col xs ={0} md={6}></Col>
                                <Col xs={12} md={6}>
                                    <p className = "number">02</p>
                                    <h2>LEARNERS</h2>
                                    <p>Know the basics but want to improve? VSM enables Learners to gain an even
                                    better understanding of the markets.Build your portfolio and your knowledge base
                                    with zero risk.Boost your learning by following peers and try your hand at
                                    simulations.Get expert insight and an edge in the job market. 
                                    </p>
                                </Col>
                            </Row>
                            <hr/>
                            <Row>
                                <Col xs={12} md={6}>
                                    <p className = "number">03</p>
                                    <h2>PRO INVESTOR</h2>
                                    <p>
                                        VSM gives Pro Investors access to real market data from multiple global
                                        exchanges to trade in.Manage your portfolio like an expert,test your investment
                                        strategies,leverage analytics and compete in simulations against other
                                        top traders to earn great prizes including job opportunities.
                                    </p>
                                </Col>
                                <Col xs ={0} md={6}></Col>
                            </Row>
                            <hr/>
                            <Row>
                                <Col className = "colText" xs ={12}>
                                    <a href="/signUp"><Button label = "START FOR FREE"/></a>
                                </Col>
                            </Row>
                        </Container>
                    </div>    
                    <div className="footer">
                        <Container>
                            <Row>
                                <Col xs={12} md = {6}>
                                    <p>VSM</p>
                                    <p>Learn.Share.Earn</p>
                                    <br/>
                                    <p>Access a global trading resource anytime, anywhere. VSM helps you learn, develop and improve your investing skills.</p> 
                                </Col>
                                <Col xs={12} md={6}>
                                    <div className="colText">
                                        <ul>
                                            <li>COMPANY</li>
                                            <br/>
                                            <li>ABOUT US</li>
                                            <li>BLOG</li>
                                            <li>HELP CENTER</li>
                                            <li>TERMS OF USE</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                            <Row>
                                <Col xs ={12}>
                                    <p>Mumbai Office: VJTI , H R Mahajani Rd, Matunga, Mumbai, Maharashtra 400019	/	Email: info@vsm.com</p>
                                    <p>© 2015-2016 VSM. All Right Reserved.</p>
                                </Col>
                            </Row>
                        </Container>
                    </div>
            </div>
        );
    }
}